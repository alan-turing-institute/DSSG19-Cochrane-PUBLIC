drop table if exists raw.reviews_studies;

create table raw.reviews_studies(
	cn varchar,
	study_id varchar,
	study_type varchar
);

\copy raw.reviews_studies from '/data/raw/review_studies.csv' DELIMITER ',' HEADER CSV;
