--turns all col names in entire db into lower case
-- makes it so that only rows are shown (without column information)
\t on
--baseline script which will be updated with schema, table and col names
select 'ALTER TABLE '||'"'||raw.papers||'"'||' RENAME COLUMN '||'"'||column_name||'"'||' TO ' || lower(column_name)||';'
-- pulls all column names where not lower case
-- a list of alter / rename commands is generated
from information_schema.columns
where table_schema = 'public' and lower(column_name) != column_name
-- executes 
\gexec
