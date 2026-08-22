-- reset.sql(開發用,正式環境別跑)
-- 連 schema_migrations 一起清掉,下次 migrate 才會從頭重跑
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS schema_migrations CASCADE;
