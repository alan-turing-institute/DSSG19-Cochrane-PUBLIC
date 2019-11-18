create table if not exists semantic.reviews (i int);
truncate semantic.reviews;
drop table if exists semantic.reviews;
create table semantic.reviews as(
	select a.cn,
       	a.last_modified_date,
	a.review_group,
	a.title,
	a.abstract_background,
	a.abstract_objectives,
	a.abstract_selection_criteria,
	a.abstract_data_collection
	from clean.reviews a
	inner join
	(select cn, max(last_modified_date) as most_recent_date
	from clean.reviews group by cn) b
	on a.cn = b.cn and a.last_modified_date = b.most_recent_date
);
alter table semantic.reviews
add primary key (cn);
