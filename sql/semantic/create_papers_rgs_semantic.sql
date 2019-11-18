--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists semantic;
create table if not exists semantic.papers_rgs (i int);
truncate semantic.papers_rgs;
drop table if exists semantic.papers_rgs;

--creates unique links for papers and reviews
--we've checked to ensure that values of interest are distinct
drop table if exists tmp_papers_rgs;
create temporary table tmp_papers_rgs as(select recordid, inregister from clean.papers where inregister != 'NULL');
create table semantic.papers_rgs as(select distinct * from tmp_papers_rgs);
