from flask import Flask, render_template, request, jsonify
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

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
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        # Use the secret file path provided by Render
        creds_path = "/etc/secrets/credentials.json"
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet by URL
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1b4_yuhEeLN-u21KHLEJOenDa1EdG6iQXpUi8ASXYZHk/edit?gid=1170846120"
        ).worksheet("Raw Data")  # Open Raw Data Sheet
        data = sheet.get_all_records()

        # Convert to DataFrame and clean column names
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names
        print("✅ Successfully loaded sheet data.")
        print("📊 Columns found:", df.columns.tolist())

        # Cache the data
        cached_data = df
        return df

    except Exception as e:
        print(f"❌ Error loading Google Sheets data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    try:
        df = get_sheet_data()
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
                            'publisher joined date', 'revenue', 'order/action', 'clicks', 'pub category',
                            'advertiser type', 'age rank', 'moz rank', 'domain authority', 'region', 'url',
                            'contact name', 'email', 'work', 'cell']
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
                'order/action': 'sum',
                'clicks': 'sum'
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
                                        'moz rank', 'domain authority', 'region', 'url', 'contact name',
                                        'email', 'work', 'cell', 'affiliate brand']].to_dict(orient='records')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Bind to all network interfaces