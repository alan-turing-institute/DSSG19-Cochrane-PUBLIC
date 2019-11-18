--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists semantic;
create table if not exists semantic.papers_reviews (i int);
truncate semantic.papers_reviews;
drop table if exists semantic.papers_reviews;

--creates unique links for papers and reviews
--we've checked to ensure that values of interest are distinct
drop table if exists tmp_papers_reviews;
create temporary table tmp_papers_reviews as(select recordid, cn, sid from clean.papers where cn != 'NULL');
create table semantic.papers_reviews as(select distinct * from tmp_papers_reviews);
