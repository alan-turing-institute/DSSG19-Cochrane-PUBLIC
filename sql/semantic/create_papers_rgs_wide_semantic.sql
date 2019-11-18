--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists semantic;
create table if not exists semantic.papers_rgs_wide (i int);
truncate semantic.papers_rgs_wide;
drop table if exists semantic.papers_rgs_wide;

--changes papers_rgs into wide format
alter table if exists semantic.papers_rgs_wide add primary key (recordid);
