import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

# Set the URL and headers for the Yahoo Finance page
url = "https://finance.yahoo.com/quote/AAPL/history?p=AAPL"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

# Use requests to get the page content and BeautifulSoup to parse it
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.content, "html.parser")

# Use BeautifulSoup to find the historical price table and extract the data
table = soup.find("table", {"data-test": "historical-prices"})
df = pd.read_html(str(table))[0]

df.to_csv("HistData.csv")
# Print the DataFrame of historical price data
print(df.head)
