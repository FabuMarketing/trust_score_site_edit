from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import csv
import io
from fpdf import FPDF  # For PDF generation

# Create the Flask application instance
app = Flask(__name__)

# Global variable to cache Google Sheets data
cached_data = None

# Load data from Google Sheets
def get_sheet_data():
    global cached_data
    if cached_data is not None:
        print("✅ Using cached data.")
        return cached_data

    try:
        # Define the scope and credentials for Google Sheets API
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet by URL
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1b4_yuhEeLN-u21KHLEJOenDa1EdG6iQXpUi8ASXYZHk/edit?gid=1170846120"
        ).worksheet("Raw Data")
        data = sheet.get_all_records()

        # Convert to DataFrame and clean column names
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip().str.lower()
        print("✅ Successfully loaded sheet data.")
        print("📊 Columns found:", df.columns.tolist())

        # Rename columns to match the required names
        column_mapping = {
            'avg. ahrefs dr': 'ahrefs dr',
            'avg. semrush as': 'semrush dr',
            'page authority': 'moz dr',
            'column aq': 'cr',  # Map Column AQ to 'cr'
            'order/action': 'order active'  # Rename Order/Action to Order Active
        }
        df.rename(columns=column_mapping, inplace=True)

        # Clean and convert numeric columns
        numeric_columns = ['trust score rating', 'revenue', 'order active', 'clicks', 'ahrefs dr', 'semrush dr', 'moz dr', 'cr']
        for col in numeric_columns:
            if col in df.columns:
                print(f"🔍 Cleaning column: {col}")
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        cached_data = df
        return df

    except Exception as e:
        print(f"❌ Error loading Google Sheets data: {e}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    try:
        df = get_sheet_data()
        if df.empty:
            return jsonify([])

        partner_names = df['partner name'].dropna().unique().tolist()
        query = request.args.get('query', '').lower()
        suggestions = [name for name in partner_names if query in name.lower()]
        return jsonify(suggestions)
    except Exception as e:
        print(f"❌ Error in autocomplete: {e}")
        return jsonify([])

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get user input from the form
        partner_input = request.form['Partner Name'].strip().lower()
        print(f"🔎 User input: {partner_input}")

        # Load the data from Google Sheets
        df = get_sheet_data()

        if df.empty:
            return render_template('result.html', error="Unable to load data from Google Sheets.")

        # Validate required columns
        required_columns = ['partner name', 'trust score rating', 'affiliate brand', 'program joined date',
                            'publisher joined date', 'revenue', 'order active', 'clicks', 'pub category',
                            'advertiser type', 'age rank', 'ahrefs dr', 'semrush dr', 'moz dr', 'url',
                            'contact name', 'email', 'work', 'cell', 'cr']  # Include 'cr'
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Missing column: {col}")
                return render_template('result.html', error=f"Missing column: {col}")

        # Normalize partner names for comparison
        df['partner name normalized'] = df['partner name'].str.strip().str.lower()

        # Check if partner exists
        if partner_input not in df['partner name normalized'].values:
            print("⚠️ Partner not found in the dataset.")
            return render_template('result.html', error="Partner Name not found.")

        # Calculate average Trust Score Rating for each partner
        df_avg = df.groupby('partner name', as_index=False).agg({
            'trust score rating': 'mean'
        }).rename(columns={'trust score rating': 'avg trust score rating'})

        # Merge the average trust score back into the original DataFrame
        df = df.merge(df_avg, on='partner name', how='left')

        # Get selected partner details
        selected = df[df['partner name normalized'] == partner_input]
        trust_score = selected['avg trust score rating'].iloc[0]

        # Aggregate affiliate brand data
        affiliate_brands = (
            selected.groupby(['affiliate brand', 'program joined date', 'publisher joined date'], as_index=False)
            .agg({
                'revenue': 'sum',
                'order active': 'sum',
                'clicks': 'sum',
                'cr': 'mean'  # Include CR as the average
            })
            .assign(
                revenue=lambda x: x['revenue'].apply(lambda v: "Yes" if v > 0 else "No"),
                action_active=lambda x: x['order active'].apply(lambda v: "Yes" if v > 0 else "No"),  # Fix Action Active
                clicks=lambda x: x['clicks'].apply(lambda v: "Yes" if v > 0 else "No"),
                cr=lambda x: x['cr']  # Display the actual CR value from the sheet
            )
            .rename(columns={
                'revenue': 'Revenue Active',
                'action_active': 'Action Active',  # Rename for display
                'clicks': 'Click Active',
                'cr': 'CR'
            })
            .to_dict(orient='records')
        )
        print(f"✅ Found partner: {selected['partner name'].iloc[0]} with Trust Score: {trust_score}")

        # Find top 10 most similar partners
        df['score_diff'] = abs(df['avg trust score rating'] - trust_score)
        top_similar = df[(df['pub category'] == selected['pub category'].iloc[0]) &
                         (df['advertiser type'] == selected['advertiser type'].iloc[0])].drop_duplicates(
            subset=['partner name']).sort_values(by='score_diff').head(10)

        # Prepare data for the template
        similar_partners = top_similar[['partner name', 'avg trust score rating', 'age rank',
                                        'ahrefs dr', 'semrush dr', 'moz dr', 'url', 'contact name',
                                        'email', 'work', 'cell', 'pub category', 'advertiser type']].to_dict(orient='records')

        return render_template(
            'result.html',
            partner_name=selected['partner name'].iloc[0],
            trust_score=trust_score,
            affiliate_brands=affiliate_brands,
            similar_partners=similar_partners
        )

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return render_template('result.html', error="An unexpected error occurred.")

@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        affiliate_brands = request.form.get('affiliate_brands', '[]')
        similar_partners = request.form.get('similar_partners', '[]')

        affiliate_brands = eval(affiliate_brands) if affiliate_brands else []
        similar_partners = eval(similar_partners) if similar_partners else []

        output = io.StringIO()
        writer = csv.writer(output)

        if affiliate_brands:
            writer.writerow(['Affiliate Brands'])
            writer.writerow(['Affiliate Brand', 'Program Joined Date', 'Publisher Joined Date', 'Revenue Active', 'Action Active', 'Click Active', 'CR'])
            for brand in affiliate_brands:
                writer.writerow([
                    brand['affiliate brand'],
                    brand['program joined date'],
                    brand['publisher joined date'],
                    brand['Revenue Active'],
                    brand['Action Active'],
                    brand['Click Active'],
                    brand['CR']
                ])
            writer.writerow([])

        if similar_partners:
            writer.writerow(['Top Similar Partners'])
            writer.writerow(['Partner Name', 'Trust Score', 'Age Rank', 'Ahrefs DR', 'Semrush DR', 'Moz DR', 'URL', 'Contact Name', 'Email', 'Work', 'Cell', 'Pub Category', 'Advertiser Type'])
            for partner in similar_partners:
                writer.writerow([
                    partner['partner name'],
                    partner['avg trust score rating'],
                    partner['age rank'],
                    partner['ahrefs dr'],
                    partner['semrush dr'],
                    partner['moz dr'],
                    partner['url'],
                    partner['contact name'],
                    partner['email'],
                    partner['work'],
                    partner['cell'],
                    partner['pub category'],
                    partner['advertiser type']
                ])

        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    except Exception as e:
        print(f"❌ Error generating CSV: {e}")
        return "Error generating CSV", 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        affiliate_brands = request.form.get('affiliate_brands', '[]')
        similar_partners = request.form.get('similar_partners', '[]')

        affiliate_brands = eval(affiliate_brands) if affiliate_brands else []
        similar_partners = eval(similar_partners) if similar_partners else []

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        if affiliate_brands:
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Affiliate Brands", ln=True, align="L")
            pdf.set_font("Arial", size=12)
            for brand in affiliate_brands:
                pdf.cell(200, 10, txt=f"{brand['affiliate brand']} - CR: {brand['CR']}", ln=True, align="L")

        if similar_partners:
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Top Similar Partners", ln=True, align="L")
            pdf.set_font("Arial", size=12)
            for partner in similar_partners:
                pdf.cell(200, 10, txt=f"{partner['partner name']} - Pub Category: {partner['pub category']}, Advertiser Type: {partner['advertiser type']}", ln=True, align="L")

        response = make_response(pdf.output(dest='S').encode('latin1'))
        response.headers['Content-Disposition'] = 'attachment; filename=result.pdf'
        response.headers['Content-Type'] = 'application/pdf'
        return response

    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return "Error generating PDF", 500

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)