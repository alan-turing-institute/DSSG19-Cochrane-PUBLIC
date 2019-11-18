drop table if exists raw.reviews_meta;

create table raw.reviews_meta(
  cd_number varchar,
  review_title varchar,
  review_group varchar,
  publication_flag varchar,
  placeholder varchar
);

\copy raw.reviews_meta from '/data/published_reviews_quoted.csv' DELIMITER ',' CSV HEADER;
