"""
company_lookup.py
Static mapping of popular company names to Yahoo Finance ticker symbols.
Used to power a dropdown so users don't need to know exact ticker syntax.
Any ticker not in this list can still be typed directly (e.g. "TCS.NS").
"""

NSE_COMPANIES = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro (L&T)": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Oil & Natural Gas Corp (ONGC)": "ONGC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
}


def get_ticker(company_name: str) -> str:
    """Look up a ticker by exact company name. Returns None if not found."""
    return NSE_COMPANIES.get(company_name)


def get_company_list() -> list:
    """Return sorted list of (company_name, ticker) pairs for populating a dropdown."""
    return sorted(NSE_COMPANIES.items())