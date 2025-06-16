from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import io


# Create the Flask application instance
app = Flask(__name__)


# Global variable to cache Google Sheets data
cached_data = None


# Load data from Google Sheets
def get_sheet_data():
   global cached_data
   if cached_data is not None:
       print(":white_check_mark: Using cached data.")
       return cached_data
   try:
       scope = ["https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"]
       creds_path = "credentials.json"  # Update this path if necessary
       creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
       client = gspread.authorize(creds)
       sheet = client.open_by_url(
           "https://docs.google.com/spreadsheets/d/1b4_yuhEeLN-u21KHLEJOenDa1EdG6iQXpUi8ASXYZHk/edit?gid=1170846120"
       ).worksheet("Raw Data")  # Open Raw Data Sheet
       data = sheet.get_all_records()
       df = pd.DataFrame(data)
       df.columns = df.columns.str.strip().str.lower()  # Normalize column names
       print(":white_check_mark: Successfully loaded sheet data.")
       print(":bar_chart: Columns found in DataFrame:", df.columns.tolist())  # Debugging statement
       column_mapping = {
           'avg. ahrefs dr': 'ahrefs dr',
           'avg. semrush as': 'semrush dr',
           'page authority': 'moz dr',
           'trust score rating': 'trust score rating'  # Ensure correct column name
       }
       df.rename(columns=column_mapping, inplace=True)
       numeric_columns = ['trust score rating', 'revenue', 'order/action', 'clicks', 'ahrefs dr', 'semrush dr', 'moz dr']
       for col in numeric_columns:
           if col in df.columns:
               df[col] = pd.to_numeric(df[col], errors='coerce')
               df[col].fillna(0, inplace=True)
       cached_data = df
       return df
   except Exception as e:
       print(f":x: Error loading Google Sheets data: {e}")
       return pd.DataFrame()


# Extract content from URL
def extract_content_from_url(url):
   try:
       response = requests.get(url, timeout=10)
       response.raise_for_status()  # Raise an error for bad responses
       soup = BeautifulSoup(response.text, 'html.parser')
      
       # Extract meta tags and visible text
       meta_tags = ' '.join([tag.get('content', '') for tag in soup.find_all('meta')])
       visible_text = ' '.join([text.strip() for text in soup.stripped_strings])
      
       return meta_tags + ' ' + visible_text
   except Exception as e:
       print(f":x: Error extracting content from URL: {e}")
       return ""


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
       print(f":x: Error in autocomplete: {e}")
       return jsonify([])


@app.route('/result', methods=['POST'])
def result():
   try:
       # Get user input from the form
       partner_input = request.form['Partner Name'].strip().lower()
       print(f":mag_right: User input: {partner_input}")
      
       # Load the data from Google Sheets
       df = get_sheet_data()
       if df.empty:
           return render_template('result_trust_score.html', error="Unable to load data from Google Sheets.")
      
       # Normalize partner names for comparison
       df['partner name normalized'] = df['partner name'].str.strip().str.lower()
      
       # Check if partner exists
       if partner_input not in df['partner name normalized'].values:
           print(":warning: Partner not found in the dataset.")
           return render_template('result_trust_score.html', error="Partner Name not found.")
      
       # Filter data specific to the selected Partner Name
       selected = df[df['partner name normalized'] == partner_input]
       if selected.empty:
           return render_template('result_trust_score.html', error="No data found for the selected Partner Name.")
      
       # Calculate Trust Score Rating
       trust_score = selected['trust score rating'].mean()
      
       # Aggregate affiliate brand data specific to the selected Partner Name
       affiliate_brands = (
           selected.groupby(['affiliate brand', 'program joined date', 'publisher joined date'], as_index=False)
           .agg({'revenue': 'sum', 'order/action': 'sum', 'clicks': 'sum'})
       )
      
       # Compute CR% as ('Order/Action' / 'Clicks') * 100
       affiliate_brands['cr'] = affiliate_brands.apply(
           lambda row: (row['order/action'] / row['clicks'] * 100) if row['clicks'] > 0 else 0, axis=1
       )
       affiliate_brands = affiliate_brands.to_dict(orient='records')
      
       # Find top 10 most similar partners
       df['score_diff'] = abs(df['trust score rating'] - trust_score)
       top_similar = df[(df['pub category'] == selected['pub category'].iloc[0]) &
                        (df['advertiser type'] == selected['advertiser type'].iloc[0])].drop_duplicates(
           subset=['partner name']).sort_values(by='score_diff').head(10)
      
       # Prepare data for the template
       similar_partners = top_similar[['partner name', 'trust score rating', 'age rank',
                                       'ahrefs dr', 'semrush dr', 'moz dr', 'url', 'contact name',
                                       'email', 'work', 'cell', 'affiliate brand', 'pub category', 'advertiser type']].to_dict(orient='records')
      
       return render_template(
           'result_trust_score.html',
           partner_name=selected['partner name'].iloc[0],
           trust_score=trust_score,  # Pass trust_score to the template
           affiliate_brands=affiliate_brands,
           similar_partners=similar_partners
       )
   except Exception as e:
       print(f":x: Error processing request: {e}")
       return render_template('result_trust_score.html', error="An unexpected error occurred.")


@app.route('/publisher_lookup', methods=['POST'])
def publisher_lookup():
   try:
       brand_url = request.form['Brand URL'].strip().lower()
       print(f":mag_right: Scraping content from URL: {brand_url}")
      
       # Extract content from the URL
       url_content = extract_content_from_url(brand_url)
       if not url_content:
           return render_template('result_brand_lookup.html', error="Unable to extract content from the URL.")
      
       # Load the data from Google Sheets
       df = get_sheet_data()
       if df.empty:
           return render_template('result_brand_lookup.html', error="Unable to load data from Google Sheets.")
      
       # Prepare data for matching
       vectorizer = TfidfVectorizer(stop_words='english')
       pub_categories = df['pub category'].dropna().unique().tolist()
       advertiser_types = df['advertiser type'].dropna().unique().tolist()
      
       # Combine Pub Categories, Advertiser Types, and scraped content
       combined_data = pub_categories + advertiser_types + [url_content]
       tfidf_matrix = vectorizer.fit_transform(combined_data)
      
       # Calculate similarity scores
       similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
       best_match_index = similarity_scores.argmax()
      
       # Determine the best match
       if best_match_index < len(pub_categories):
           matched_category = pub_categories[best_match_index]
           matched_type = None
       else:
           matched_category = None
           matched_type = advertiser_types[best_match_index - len(pub_categories)]
      
       print(f":white_check_mark: Matched Pub Category: {matched_category}, Matched Advertiser Type: {matched_type}")
      
       # Filter publishers based on the matched category and type
       filtered_publishers = df[
           (df['pub category'] == matched_category) | (df['advertiser type'] == matched_type)
       ].drop_duplicates(subset=['partner name']).sort_values(by='trust score rating', ascending=False)
      
       # Ensure at least 10 publishers are displayed
       if len(filtered_publishers) < 10:
           additional_publishers = df.sort_values(by='trust score rating', ascending=False).head(10 - len(filtered_publishers))
           filtered_publishers = pd.concat([filtered_publishers, additional_publishers]).drop_duplicates(subset=['partner name']).head(10)
      
       # Prepare data for the template
       publishers = filtered_publishers[['partner name', 'trust score rating', 'age rank',
                                         'ahrefs dr', 'semrush dr', 'moz dr', 'url', 'contact name',
                                         'email', 'work', 'cell', 'pub category', 'advertiser type']].to_dict(orient='records')
      
       return render_template(
           'result_brand_lookup.html',
           publishers=publishers
       )
   except Exception as e:
       print(f":x: Error processing request: {e}")
       return render_template('result_brand_lookup.html', error="An unexpected error occurred.")


@app.route('/download_csv', methods=['POST'])
def download_csv():
   try:
       publishers = request.json.get('publishers', [])
       output = io.StringIO()
       writer = csv.DictWriter(output, fieldnames=publishers[0].keys())
       writer.writeheader()
       writer.writerows(publishers)
       response = make_response(output.getvalue())
       response.headers["Content-Disposition"] = "attachment; filename=publishers.csv"
       response.headers["Content-Type"] = "text/csv"
       return response
   except Exception as e:
       print(f":x: Error generating CSV: {e}")
       return jsonify({"error": "Failed to generate CSV"}), 500


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
   # Placeholder for PDF generation logic
   return jsonify({"error": "PDF generation not implemented yet"}), 501


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)

