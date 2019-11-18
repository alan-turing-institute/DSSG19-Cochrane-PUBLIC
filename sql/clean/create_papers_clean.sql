--pipeline for cleaning the papers db
--all sql code from here on out requires that
--all column names are in lower case, a simple
--script for this can be found in sql/utils

--admin
create schema if not exists clean;
drop table if exists clean.papers;

--functions
create or replace function yr_clean(arg text)
returns int language plpgsql as $$
begin
	begin
		return cast(left(arg, 4) as int);
	exception when others then return null;
	end;
end $$;

--create new table from raw and create var
create table clean.papers as(
select *,
case when yr_clean(yr) between 1930 and 2019 then 1 else 0 end as yr_valid
from raw.papers);

--drop unnecessary columns
alter table clean.papers
drop column recordtype,
drop column pg,
drop column no,
drop column ot,
drop column searchtext,
drop column crg,
drop column vl,
drop column clean_title,
drop column clean_abstract,
drop column short_abstract,
drop column short_title,
drop column include;

--cleaning some table variables
update clean.papers
set au = replace(au, '//', ' '),
ad = replace(ad, '//', ' '),
cc = replace(cc, '//', ' '),
mc = replace(mc, '//', ' '),
mh = replace(mh, '//', ' '),
pt = replace(pt, '//', ' ');

--clean up issues where paper in review but no review group
create temporary table tmp_papers2 as(with tbl as (select cn, review_group from raw.reviews),
tbl2 as (select recordid, ti, au, so, yr, ab, ad, cc, doi,
em, la, mc, mh, pm, pt, centralid, isnct, tr,
sid, yr_valid, clean.papers.cn, inregister, tbl.review_group
from clean.papers left join tbl on clean.papers.cn = tbl.cn)
select *,
case when cn!='NULL' and inregister='NULL' then review_group else inregister end as final_register
from tbl2);

alter table tmp_papers2
drop column inregister,
drop column review_group;

alter table tmp_papers2
rename column final_register to inregister;

delete from tmp_papers2
	where inregister in ('STD',''); 

drop table if exists clean.papers;
create table clean.papers as table tmp_papers2;

--final commit to close issue
