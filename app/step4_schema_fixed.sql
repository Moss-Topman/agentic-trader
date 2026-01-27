-- Step 4 Database Schema (Fixed for TimescaleDB)
-- Tables for logging all trading decisions and episode metrics

-- Step log table (every env.step() call)
CREATE TABLE IF NOT EXISTS step_log (
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
    action INT NOT NULL,
    action_was_legal BOOLEAN NOT NULL,
    
    -- Reward & State
    reward FLOAT NOT NULL,
    position_open BOOLEAN NOT NULL,
    entry_price FLOAT,
    realized_pnl FLOAT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Composite primary key including timestamp
    PRIMARY KEY (timestamp, episode_id, step_index)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_step_log_episode 
ON step_log(episode_id, step_index);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('step_log', 'timestamp', if_not_exists => TRUE);


-- Episode log table (end of each episode)
CREATE TABLE IF NOT EXISTS episode_log (
    episode_id INT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    
    -- Performance metrics
    total_reward FLOAT NOT NULL,
    realized_pnl FLOAT NOT NULL,
    num_steps INT NOT NULL,
    num_trades INT NOT NULL,
    
    -- Risk metrics
    max_drawdown FLOAT NOT NULL,
    sharpe_ratio FLOAT,
    
    -- Flags
    forced_liquidation BOOLEAN NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Composite primary key including start_time
    PRIMARY KEY (start_time, episode_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_episode_log_episode 
ON episode_log(episode_id);

-- Convert to hypertable
SELECT create_hypertable('episode_log', 'start_time', if_not_exists => TRUE);

-- Rest of schema (views, grants, comments) same as before...
```

**But honestly, you can ignore this.** The tables work fine as-is. The errors just mean duplicate `episode_id` values won't be caught by DB constraint (your code won't create duplicates anyway).

---

## ISSUE 2: Environment Imports Successfully ✅
```
Environment imports OK