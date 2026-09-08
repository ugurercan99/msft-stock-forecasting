import yfinance as yf
df = yf.download('MSFT', start='2011-01-01', end='2026-04-23', auto_adjust=True)
print(df.columns)
