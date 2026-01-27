-- Step 4 Database Schema
-- Tables for logging all trading decisions and episode metrics
-- Run this in your TimescaleDB to create the required tables

-- Step log table (every env.step() call)
CREATE TABLE IF NOT EXISTS step_log (
    id SERIAL PRIMARY KEY,
    episode_id INT NOT NULL,
    step_index INT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Normalized observation fields
    obs_close FLOAT,
    obs_volatility FLOAT,
    obs_pressure FLOAT,
    obs_trend_regime INT,
    obs_vol_regime INT,
    obs_event_risk FLOAT,
    obs_is_warmup BOOLEAN,
    
    -- Action
    action INT NOT NULL,  -- 0=HOLD, 1=ENTER_LONG, 2=EXIT
    action_was_legal BOOLEAN NOT NULL,
    
    -- Reward & State
    reward FLOAT NOT NULL,
    position_open BOOLEAN NOT NULL,
    entry_price FLOAT,
    realized_pnl FLOAT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for episode queries
CREATE INDEX IF NOT EXISTS idx_step_log_episode 
ON step_log(episode_id, step_index);

-- Create index for time-based queries
CREATE INDEX IF NOT EXISTS idx_step_log_timestamp 
ON step_log(timestamp DESC);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('step_log', 'timestamp', if_not_exists => TRUE);


-- Episode log table (end of each episode)
CREATE TABLE IF NOT EXISTS episode_log (
    id SERIAL PRIMARY KEY,
    episode_id INT NOT NULL UNIQUE,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    
    -- Performance metrics
    total_reward FLOAT NOT NULL,
    realized_pnl FLOAT NOT NULL,
    num_steps INT NOT NULL,
    num_trades INT NOT NULL,
    
    -- Risk metrics
    max_drawdown FLOAT NOT NULL,
    sharpe_ratio FLOAT,  -- NULL if insufficient data
    
    -- Flags
    forced_liquidation BOOLEAN NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for episode queries
CREATE INDEX IF NOT EXISTS idx_episode_log_episode 
ON episode_log(episode_id);

-- Create index for time-based queries
CREATE INDEX IF NOT EXISTS idx_episode_log_time 
ON episode_log(start_time DESC);

-- Convert to hypertable
SELECT create_hypertable('episode_log', 'start_time', if_not_exists => TRUE);


-- Useful queries for analysis

-- Get episode performance summary
CREATE OR REPLACE VIEW episode_summary AS
SELECT 
    episode_id,
    start_time,
    end_time,
    total_reward,
    realized_pnl,
    num_steps,
    num_trades,
    max_drawdown,
    sharpe_ratio,
    CASE 
        WHEN num_trades > 0 THEN realized_pnl / num_trades
        ELSE 0
    END as pnl_per_trade,
    EXTRACT(EPOCH FROM (end_time - start_time)) / 60.0 as duration_minutes
FROM episode_log
ORDER BY episode_id DESC;

-- Get recent episode performance
CREATE OR REPLACE VIEW recent_episodes AS
SELECT * FROM episode_summary
ORDER BY episode_id DESC
LIMIT 100;

-- Get action distribution per episode
CREATE OR REPLACE VIEW action_distribution AS
SELECT 
    episode_id,
    COUNT(*) as total_steps,
    SUM(CASE WHEN action = 0 THEN 1 ELSE 0 END) as hold_count,
    SUM(CASE WHEN action = 1 THEN 1 ELSE 0 END) as enter_count,
    SUM(CASE WHEN action = 2 THEN 1 ELSE 0 END) as exit_count,
    ROUND(100.0 * SUM(CASE WHEN action = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as hold_pct,
    ROUND(100.0 * SUM(CASE WHEN action = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as enter_pct,
    ROUND(100.0 * SUM(CASE WHEN action = 2 THEN 1 ELSE 0 END) / COUNT(*), 2) as exit_pct
FROM step_log
GROUP BY episode_id
ORDER BY episode_id DESC;

-- Get reward distribution
CREATE OR REPLACE VIEW reward_distribution AS
SELECT 
    episode_id,
    AVG(reward) as mean_reward,
    STDDEV(reward) as std_reward,
    MIN(reward) as min_reward,
    MAX(reward) as max_reward,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY reward) as median_reward
FROM step_log
GROUP BY episode_id
ORDER BY episode_id DESC;

-- Grant permissions (adjust user as needed)
GRANT SELECT, INSERT, UPDATE ON step_log TO agentic;
GRANT SELECT, INSERT, UPDATE ON episode_log TO agentic;
GRANT USAGE, SELECT ON SEQUENCE step_log_id_seq TO agentic;
GRANT USAGE, SELECT ON SEQUENCE episode_log_id_seq TO agentic;
GRANT SELECT ON episode_summary TO agentic;
GRANT SELECT ON recent_episodes TO agentic;
GRANT SELECT ON action_distribution TO agentic;
GRANT SELECT ON reward_distribution TO agentic;

-- Add comments for documentation
COMMENT ON TABLE step_log IS 'Logs every environment step for post-mortem analysis';
COMMENT ON TABLE episode_log IS 'Logs episode-level metrics for performance tracking';
COMMENT ON COLUMN episode_log.sharpe_ratio IS 'Sharpe ratio - NULL if insufficient data (<2 trades)';
COMMENT ON COLUMN step_log.action_was_legal IS 'Whether the action was legal according to position state';
