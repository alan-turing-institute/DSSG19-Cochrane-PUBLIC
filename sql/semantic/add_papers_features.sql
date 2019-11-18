--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists semantic;
create table if not exists semantic.papers_features (i int);
truncate semantic.papers_features;
drop table if exists semantic.papers_features;

create table semantic.papers_features as
  select a.*, b.tokens from semantic.papers a
  left join semantic.features b
  on a.recordid=b.recordid
;

alter table if exists semantic.papers_features add primary key (recordid);
