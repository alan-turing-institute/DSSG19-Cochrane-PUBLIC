--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists semantic;
create table if not exists semantic.papers (i int);
truncate semantic.papers;
drop table if exists semantic.papers;

--the semantic papers are unique records
--we've checked to ensure that values of interest are distinct
drop table if exists tmp_papers;
create temporary table tmp_papers as table clean.papers;
alter table tmp_papers drop cn, drop sid, drop inregister;
create table semantic.papers as(select distinct * from tmp_papers);
