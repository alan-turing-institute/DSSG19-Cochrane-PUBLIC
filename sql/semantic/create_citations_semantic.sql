--copy over to clean
create table if not exists semantic.citations (i int);
truncate semantic.citations;
drop table if exists semantic.citations;
create table semantic.citations as table clean.citations;

create table if not exists semantic.recordid_paperid (i int);
  truncate semantic.recordid_paperid;
  drop table if exists semantic.recordid_paperid;
  create temporary table recordid_paperid as table clean.recordid_paperid order by  random();
  alter table recordid_paperid add id serial;
  delete from recordid_paperid a using recordid_paperid b
  where a.id < b.id and a.recordid = b.recordid;
  delete from recordid_paperid a using recordid_paperid b
  where a.id < b.id and a.paperid = b.paperid;
  create table semantic.recordid_paperid as table recordid_paperid;
