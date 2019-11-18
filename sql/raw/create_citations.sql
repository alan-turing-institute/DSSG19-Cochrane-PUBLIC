--load citations edge list and titles
drop table if exists raw.citations;
create table raw.citations (
paperid varchar,
refpaperid varchar);

drop table if exists raw.recordid_paperid;
create table raw.recordid_paperid (
recordid varchar,
paperid varchar);


--no longer pulling in the citation text
--drop table if exists raw.citations_text;
--create table raw.citations_text (
--paperid varchar,
--title varchar,
--abstract varchar);


--psql code
--\copy raw.citations from '/data/citations/TuringCitations.csv' HEADER CSV;
--\copy raw.recordid_paperid from '/data/citations/TuringCRSPMRecords.csv' header csv;
--\copy raw.citations_text from '/data/citations/TuringCitationText.csv' csv;
