drop table if exists raw.reviews;

create table raw.reviews(
	cn varchar,
	last_modified_date varchar,
	review_group varchar,
	title varchar,
	abstract_background varchar,
	abstract_objectives varchar,
	abstract_selection_criteria varchar,
	abstract_data_collection varchar
);

-- \copy raw.reviews from '/data/reviews.csv' DELIMITER ',' HEADER CSV;
