
        -- NeuroBreak数据库模式
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT UNIQUE NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'running',
            config TEXT,
            results TEXT
        );
        
        CREATE TABLE IF NOT EXISTS jailbreak_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            success BOOLEAN DEFAULT FALSE,
            safety_score REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        
        CREATE TABLE IF NOT EXISTS mechanism_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            attention_entropy REAL,
            activation_magnitude REAL,
            gradient_norm REAL,
            analysis_data TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_jailbreak_experiment ON jailbreak_tests(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_mechanism_experiment ON mechanism_analysis(experiment_id);
        