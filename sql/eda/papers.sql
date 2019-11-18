--how many entries are in the db?
select count(*) from raw.papers;
--1428110

--how many unique papers are in the db?
with tbl as(select distinct "RecordID" from raw.papers) select count(*) from tbl;
select count(distinct "RecordID") from raw.papers; 
--978813

--how many entries are included in reviews?
with tbl as (select "RecordID", 
case when "CN"='NULL' then 0 else 1 end as used
from raw.papers
),
tbl2 as (select "RecordID", sum(used) from tbl group by "RecordID")
select case when sum>=1 then 'Included' else 'Not Included' end as status,
count(*) as n 
from tbl2
group by status;
--Included	144916
--Not Included	833897


--how many reviews are papers used in?
select "RecordID", count(*) as n from raw.papers
where "CN"!='NULL'
group by "RecordID" order by n desc;

--paper languages
select "LA", count(*) as n from raw.papers
group by 1 order by n desc;
--eng is by far the most used language followed by chi

--Review groups
select "CC", count(*) as n from raw.papers
where "CC"!='NULL'
group by 1 order by n desc limit 10;
-- COMPMED > VASC > ORAL > AIRWAYS

-- how many unique "CentralID"
select count(distinct "CentralID") from raw.papers;
--583830

--how many record types?
select "RecordType", count(*) as n from raw.papers
group by 1 order by n desc limit 10;
--all are reports

--what's the crosstab of included vs in register?
select 
case when "INREGISTER"='NULL' then 'yes' else 'no' end as in_register,
case when "INCLUDE"='1' then 'yes' else 'no' end as included,
count(*) as n 
from raw.papers
group by 1,2 order by 3 desc;
--reg   inc   n
--yes	yes	852734
--no	yes	462387
--yes	no	76481
--no	no	36508

--How many unique reviews are there?
select count(distinct "CN") from raw.papers;
--7384

--How many unique study ids are there?
select count(distinct "SID") from raw.papers;
--123152

--What's the largest number of papers in a review
--What are the number of papers per review? ** 
select "CN", count(*) as n
from raw.papers
where "CN"!='NULL'
group by 1 order by 2 desc;
--934

--Do all studies belong in review group registers?
select "INREGISTER", count(*) as n
from raw.papers
group by 1 order --Do all studies belong in review group registers?
select "INREGISTER", count(*) as n
from raw.papers
group by 1 order by 2 desc;
-- no 929215 are not in any register by 2 desc;
-- no 929215 are not in any register 

--are all reviews and registers and vice versa?
select 
case when "INREGISTER"='NULL' then 'no' else 'yes' end as in_register,
case when "CN"='NULL' then 'no' else 'yes' end as in_review,
count(*) as n 
from raw.papers
group by 1,2 order by 3 desc;

--finalize to push and merge



