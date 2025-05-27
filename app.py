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
            'column aq': 'cr'  # Map Column AQ to 'cr'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Clean and convert numeric columns
        numeric_columns = ['trust score rating', 'revenue', 'order/action', 'clicks', 'ahrefs dr', 'semrush dr', 'moz dr', 'cr']
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
                            'publisher joined date', 'revenue', 'order/action', 'clicks', 'pub category',
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

        # Get selected partner details
        selected = df[df['partner name normalized'] == partner_input]

        # Prepare data for the template
        partner_details = selected.to_dict(orient='records')
        print(f"✅ Partner details: {partner_details}")

        return render_template(
            'result.html',
            partner_name=partner_input,
            partner_details=partner_details
        )

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return render_template('result.html', error="An unexpected error occurred.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)