-- Market candle table
CREATE TABLE IF NOT EXISTS candles (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

-- Convert to Timescale hypertable
SELECT create_hypertable(
    'candles',
    'time',
    if_not_exists => TRUE
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
ON candles (symbol, time DESC);
