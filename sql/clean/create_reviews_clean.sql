drop table if exists clean.reviews;
create table clean.reviews as(
select
cast (cn as varchar) as cn,
cast (last_modified_date as date) as last_modified_date,
cast (review_group as varchar) as review_group,
cast (lower(title) as varchar) as title,
cast (lower(abstract_background) as varchar) as abstract_background,
cast (lower(abstract_objectives) as varchar) as abstract_objectives,
cast (lower(abstract_selection_criteria) as varchar) as abstract_selection_criteria,
cast (lower(abstract_data_collection) as varchar) as abstract_data_collection
from
raw.reviews
);
