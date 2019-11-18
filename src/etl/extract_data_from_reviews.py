import glob
import re

import pandas as pd
from lxml import etree


def create_reviews_tables(reviews_dir):
    """
    Parses Cochrane review .xml files and creates two pd.DataFrames:
    1. reviews_df : metadata about each review, including title and abstract
    2. reviews_studies_df : studies that are referenced in each review

    Parameters
    ==========
    reviews_dir : str
        Path to directory with review files.

    Returns
    =======
    reviews_df : pd.DataFrame
        One row = one review.
    reviews_studies_df : pd.DataFrame
        One row = one review-study combination, with a column indicating how the study was referenced.
    """

    all_files = glob.glob(reviews_dir + "*.rm5")

    studies_by_review = []
    study_types = ["included", "excluded", "awaiting", "ongoing"]
    bad_chars = ["\\n", "\n", "<P>", "<\P>", "</P>", "<SUP>", "<\SUP>", "</SUP>",
                 "<SUB>", "<\SUB>", "</SUB>", "\'", "<I>", "</I>", "<U>", "</U>",
                 "<BR/>", "&gt", "&lt", "&amp", "\\", "<B>", "</B>", "<\B>"]
    no_background = 0
    no_objectives = 0
    no_selection_criteria = 0
    no_data_collection = 0

    for file in all_files:
        
        root = etree.parse(file)

        # get review ID
        review_desc = root.xpath('/COCHRANE_REVIEW')
        review_desc_text = str(etree.tostring(review_desc[0]))

        try:
            review_cn = re.search(r'(CD|MR)\d{6}', file).group()
        except:
            print("BAD  ", file)
            pass
            # review_cn = re.search(r'(CD|MR)\d{6}', review_desc_text).group()

        # get year
        review_modified_date = re.findall("\d{4}-\d{2}-\d{2}", review_desc_text.split('MODIFIED=')[1])[0]

        # get review group
        review_group = review_desc_text.split('GROUP_ID="')[1].split('"')[0]

        # get title
        title_section = root.xpath('/COCHRANE_REVIEW/COVER_SHEET/TITLE')
        title = str(etree.tostring(title_section[0])).split(">", 1)[1].split("</TITLE>")[0].replace('\n', '')

        # get abstract background
        abstract_background_section = root.xpath('/COCHRANE_REVIEW/MAIN_TEXT/ABSTRACT/ABS_BACKGROUND')
        try:
            abstract_background = \
            str(etree.tostring(abstract_background_section[0])).split("<P>", 1)[1].split("</ABS_BACKGROUND>")[0]
        except IndexError:
            abstract_background = "NA"
            no_background += 1

        # get abstract objectives
        abstract_objectives_section = root.xpath('/COCHRANE_REVIEW/MAIN_TEXT/ABSTRACT/ABS_OBJECTIVES')
        try:
            abstract_objectives = \
            str(etree.tostring(abstract_objectives_section[0])).split('<P>', 1)[1].split('</ABS_OBJECTIVES>')[0]
        except IndexError:
            abstract_objectives = "NA"
            no_objectives += 1

        # get abstract selection criteria
        abstract_selection_criteria_section = root.xpath('/COCHRANE_REVIEW/MAIN_TEXT/ABSTRACT/ABS_SELECTION_CRITERIA')
        try:
            abstract_selection_criteria = \
            str(etree.tostring(abstract_selection_criteria_section[0])).split('<P>', 1)[1].split(
                '</ABS_SELECTION_CRITERIA>')[0]
        except IndexError:
            abstract_selection_criteria = "NA"
            no_selection_criteria += 1

        # get abstract data collection criteria
        abstract_data_collection_section = root.xpath('/COCHRANE_REVIEW/MAIN_TEXT/ABSTRACT/ABS_DATA_COLLECTION')
        try:
            abstract_data_collection = \
            str(etree.tostring(abstract_data_collection_section[0])).split('<P>', 1)[1].split('</ABS_DATA_COLLECTION>')[
                0]
        except:
            abstract_data_collection = "NA"
            no_data_collection += 1

        # clean text
        for char in bad_chars:
            title = title.replace(char, "")
            abstract_background = abstract_background.replace(char, "")
            abstract_objectives = abstract_objectives.replace(char, "")
            abstract_selection_criteria = abstract_selection_criteria.replace(char, "")
            abstract_data_collection = abstract_data_collection.replace(char, "")

        symbol_str = '&#(\d{3}|\d{4});'
        title = re.sub(symbol_str, '', title)
        abstract_background = re.sub(symbol_str, '', abstract_background)
        abstract_objectives = re.sub(symbol_str, '', abstract_objectives)
        abstract_selection_criteria = re.sub(symbol_str, '', abstract_selection_criteria)
        abstract_data_collection = re.sub(symbol_str, '', abstract_data_collection)

        title = re.sub("<A.*</A>", "", title, flags=re.DOTALL)
        abstract_background = re.sub("<A.*</A>", "", abstract_background, flags=re.DOTALL)
        abstract_objectives = re.sub("<A.*</A>", "", abstract_objectives, flags=re.DOTALL)
        abstract_selection_criteria = re.sub("<A.*</A>", "", abstract_selection_criteria, flags=re.DOTALL)
        abstract_data_collection = re.sub("<A.*</A>", "", abstract_data_collection, flags=re.DOTALL)

        # create dictionary
        studies_info = {"cn": review_cn,
                        "last_modified_date": review_modified_date,
                        "review_group": review_group,
                        "title": title,
                        "abstract_background": abstract_background,
                        "abstract_objectives": abstract_objectives,
                        "abstract_selection_criteria": abstract_selection_criteria,
                        "abstract_data_collection": abstract_data_collection}

        # get text for all relevant studies
        included_studies = root.xpath(
            '/COCHRANE_REVIEW/CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_INCLUDED_STUDIES/INCLUDED_CHAR')
        excluded_studies = root.xpath(
            '/COCHRANE_REVIEW/CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_EXCLUDED_STUDIES/EXCLUDED_CHAR')
        awaiting_studies = root.xpath(
            '/COCHRANE_REVIEW/CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_AWAITING_STUDIES/AWAITING_CHAR')
        ongoing_studies = root.xpath(
            '/COCHRANE_REVIEW/CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_ONGOING_STUDIES/ONGOING_CHAR')
        all_studies = [included_studies, excluded_studies, awaiting_studies, ongoing_studies]

        # iterate through and grab their IDs
        for studies, study_type in zip(all_studies, study_types):
            studies_info[study_type] = []
            for study in studies:
                study_text = str(etree.tostring(study))
                if "STUDY_ID" in study_text:
                    study_info = study_text.split('STUDY_ID="')[1].split('"')[0]
                    studies_info[study_type].append(study_info)
                else:
                    continue

        # save new element in list that is a dictionary with:
        # CN number, i.e., review ID,
        # last_modified_date of review,
        # review title
        # 4 different parts of the review abstract
        # 4 key-value pairs with
        # key=study_type and value=[list of studies of that type]
        studies_by_review.append(studies_info)

    print(
        "Files parsed with {} errors in locating abstract backgrounds, {} errors in locating abstract objectives, {} errors in locating selection criteria, {} errors locating data collection criteria.".format(
            no_background, no_objectives, no_selection_criteria, no_data_collection))

    # review and year
    reviews_df = pd.DataFrame(studies_by_review)[
        ["cn", "last_modified_date", "review_group", "title", "abstract_background", "abstract_objectives",
         "abstract_selection_criteria", "abstract_data_collection"]]

    # review and studies
    review_studies = []

    for review in studies_by_review:
        for study_type in study_types:
            stds = review[study_type]
            for std in stds:
                review_studies.append({"cn": review["cn"],
                                       "study_id": std,
                                       "study_type": study_type})

    review_studies_df = pd.DataFrame.from_dict(review_studies)

    return reviews_df, review_studies_df
