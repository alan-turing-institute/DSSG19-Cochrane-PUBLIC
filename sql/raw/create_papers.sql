--dont use this chunk of code as R uploads it automatically

create table papers (
RecordID varchar,
RecordType varchar,
TI varchar,
AU varchar,
SO varchar,
YR varchar,
PG varchar,
AB varchar,
AD varchar,
CC varchar,
DOI varchar,
EM varchar,
LA varchar,
MC varchar,
MH varchar,
NO varchar,
OT varchar,
PM varchar,
PT varchar,
VL varchar,
CentralID varchar,
ISNCT varchar,
TR varchar,
INREGISTER varchar,
CN varchar,
SID varchar,
SearchText varchar,
CRG varchar,
clean_title varchar,
clean_abstract varchar,
short_abstract varchar,
short_title varchar,
INCLUDE varchar);

-- run this code to create and set the schema
create schema raw;
alter table papers set schema raw;

-- run this code in psql directly s
\copy raw.papers from '/data/raw/papers/RecordNotNullPMID.txt' HEADER CSV DELIMITER E'\t';
