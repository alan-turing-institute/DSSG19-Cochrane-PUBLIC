--copy over to clean
drop table if exists clean.citations;
create table clean.citations as table raw.citations;

drop table if exists clean.recordid_paperid;
create table clean.recordid_paperid as table raw.recordid_paperid;
