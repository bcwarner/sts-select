#%%
import re

import pandas
import os

# This query represents dataset "pain_v_ppsp" for domain "person" and was generated for All of Us Registered Tier Dataset v7
dataset_person_sql = """
    SELECT
        person.person_id,
        person.gender_concept_id,
        p_gender_concept.concept_name as gender,
        person.birth_datetime as date_of_birth,
        person.race_concept_id,
        p_race_concept.concept_name as race,
        person.ethnicity_concept_id,
        p_ethnicity_concept.concept_name as ethnicity,
        person.sex_at_birth_concept_id,
        p_sex_at_birth_concept.concept_name as sex_at_birth 
    FROM
        `""" + os.environ["WORKSPACE_CDR"] + """.person` person 
    LEFT JOIN
        `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_gender_concept 
            ON person.gender_concept_id = p_gender_concept.concept_id 
    LEFT JOIN
        `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_race_concept 
            ON person.race_concept_id = p_race_concept.concept_id 
    LEFT JOIN
        `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_ethnicity_concept 
            ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id 
    LEFT JOIN
        `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_sex_at_birth_concept 
            ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id  
    WHERE
        person.PERSON_ID IN (SELECT
            distinct person_id  
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` cb_search_person  
        WHERE
            cb_search_person.person_id IN (SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                WHERE
                    (concept_id IN(SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (SELECT
                            CAST(cr.id as string) AS id       
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                        WHERE
                            concept_id IN (4150125)       
                            AND full_text LIKE '%_rank1]%'      ) a 
                            ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                            OR c.path LIKE CONCAT('%.', a.id) 
                            OR c.path LIKE CONCAT(a.id, '.%') 
                            OR c.path = a.id) 
                    WHERE
                        is_standard = 1 
                        AND is_selectable = 1) 
                    AND is_standard = 1 )) criteria 
            UNION
            DISTINCT SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                WHERE
                    (concept_id IN(SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (SELECT
                            CAST(cr.id as string) AS id       
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                        WHERE
                            concept_id IN (4004516)       
                            AND full_text LIKE '%_rank1]%'      ) a 
                            ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                            OR c.path LIKE CONCAT('%.', a.id) 
                            OR c.path LIKE CONCAT(a.id, '.%') 
                            OR c.path = a.id) 
                    WHERE
                        is_standard = 1 
                        AND is_selectable = 1) 
                    AND is_standard = 1 )) criteria ) 
            AND cb_search_person.person_id IN (SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                WHERE
                    (concept_id IN(SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (SELECT
                            CAST(cr.id as string) AS id       
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                        WHERE
                            concept_id IN (1585710, 43528895, 40192389, 1740639, 1586134, 1333342, 1741006, 1585855)       
                            AND full_text LIKE '%_rank1]%'      ) a 
                            ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                            OR c.path LIKE CONCAT('%.', a.id) 
                            OR c.path LIKE CONCAT(a.id, '.%') 
                            OR c.path = a.id) 
                    WHERE
                        is_standard = 0 
                        AND is_selectable = 1) 
                    AND is_standard = 0 )) criteria ) )"""

dataset_person_df = pandas.read_gbq(
    dataset_person_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

import pandas
import os

# This query represents dataset "pain_v_ppsp" for domain "survey" and was generated for All of Us Registered Tier Dataset v7
dataset_survey_sql = """
    SELECT
        answer.person_id,
        answer.survey_datetime,
        answer.survey,
        answer.question_concept_id,
        answer.question,
        answer.answer_concept_id,
        answer.answer,
        answer.survey_version_concept_id,
        answer.survey_version_name  
    FROM
        `""" + os.environ["WORKSPACE_CDR"] + """.ds_survey` answer    
    WHERE
        (
            answer.PERSON_ID IN (SELECT
                distinct person_id  
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` cb_search_person  
            WHERE
                cb_search_person.person_id IN (SELECT
                    criteria.person_id 
                FROM
                    (SELECT
                        DISTINCT person_id, entry_date, concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                    WHERE
                        (concept_id IN(SELECT
                            DISTINCT c.concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                        JOIN
                            (SELECT
                                CAST(cr.id as string) AS id       
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                            WHERE
                                concept_id IN (4150125)       
                                AND full_text LIKE '%_rank1]%'      ) a 
                                ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                OR c.path LIKE CONCAT('%.', a.id) 
                                OR c.path LIKE CONCAT(a.id, '.%') 
                                OR c.path = a.id) 
                        WHERE
                            is_standard = 1 
                            AND is_selectable = 1) 
                        AND is_standard = 1 )) criteria 
                UNION
                DISTINCT SELECT
                    criteria.person_id 
                FROM
                    (SELECT
                        DISTINCT person_id, entry_date, concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                    WHERE
                        (concept_id IN(SELECT
                            DISTINCT c.concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                        JOIN
                            (SELECT
                                CAST(cr.id as string) AS id       
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                            WHERE
                                concept_id IN (4004516)       
                                AND full_text LIKE '%_rank1]%'      ) a 
                                ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                OR c.path LIKE CONCAT('%.', a.id) 
                                OR c.path LIKE CONCAT(a.id, '.%') 
                                OR c.path = a.id) 
                        WHERE
                            is_standard = 1 
                            AND is_selectable = 1) 
                        AND is_standard = 1 )) criteria ) 
                AND cb_search_person.person_id IN (SELECT
                    criteria.person_id 
                FROM
                    (SELECT
                        DISTINCT person_id, entry_date, concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                    WHERE
                        (concept_id IN(SELECT
                            DISTINCT c.concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                        JOIN
                            (SELECT
                                CAST(cr.id as string) AS id       
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                            WHERE
                                concept_id IN (1585710, 43528895, 40192389, 1740639, 1586134, 1333342, 1741006, 1585855)       
                                AND full_text LIKE '%_rank1]%'      ) a 
                                ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                OR c.path LIKE CONCAT('%.', a.id) 
                                OR c.path LIKE CONCAT(a.id, '.%') 
                                OR c.path = a.id) 
                        WHERE
                            is_standard = 0 
                            AND is_selectable = 1) 
                        AND is_standard = 0 )) criteria ) )
            )"""

dataset_survey_df = pandas.read_gbq(
    dataset_survey_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

dataset_survey_df.head(5)
import pandas
import os

# This query represents dataset "pain_v_ppsp" for domain "procedure" and was generated for All of Us Registered Tier Dataset v7
dataset_procedure_sql = """
    SELECT
        procedure.person_id,
        procedure.procedure_concept_id,
        p_standard_concept.concept_name as standard_concept_name,
        p_standard_concept.concept_code as standard_concept_code,
        p_standard_concept.vocabulary_id as standard_vocabulary,
        procedure.procedure_datetime,
        procedure.procedure_type_concept_id,
        p_type.concept_name as procedure_type_concept_name,
        procedure.modifier_concept_id,
        p_modifier.concept_name as modifier_concept_name,
        procedure.quantity,
        procedure.visit_occurrence_id,
        p_visit.concept_name as visit_occurrence_concept_name,
        procedure.procedure_source_value,
        procedure.procedure_source_concept_id,
        p_source_concept.concept_name as source_concept_name,
        p_source_concept.concept_code as source_concept_code,
        p_source_concept.vocabulary_id as source_vocabulary,
        procedure.modifier_source_value 
    FROM
        ( SELECT
            * 
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.procedure_occurrence` procedure 
        WHERE
            (
                procedure_concept_id IN (SELECT
                    DISTINCT c.concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                JOIN
                    (SELECT
                        CAST(cr.id as string) AS id       
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                    WHERE
                        concept_id IN (4004516)       
                        AND full_text LIKE '%_rank1]%'      ) a 
                        ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                        OR c.path LIKE CONCAT('%.', a.id) 
                        OR c.path LIKE CONCAT(a.id, '.%') 
                        OR c.path = a.id) 
                WHERE
                    is_standard = 1 
                    AND is_selectable = 1)
            )  
            AND (
                procedure.PERSON_ID IN (SELECT
                    distinct person_id  
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` cb_search_person  
                WHERE
                    cb_search_person.person_id IN (SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (4150125)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 1 
                                AND is_selectable = 1) 
                            AND is_standard = 1 )) criteria 
                    UNION
                    DISTINCT SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (4004516)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 1 
                                AND is_selectable = 1) 
                            AND is_standard = 1 )) criteria ) 
                    AND cb_search_person.person_id IN (SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (1585710, 43528895, 40192389, 1740639, 1586134, 1333342, 1741006, 1585855)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 0 
                                AND is_selectable = 1) 
                            AND is_standard = 0 )) criteria ) )
                )
            ) procedure 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_standard_concept 
                ON procedure.procedure_concept_id = p_standard_concept.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_type 
                ON procedure.procedure_type_concept_id = p_type.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_modifier 
                ON procedure.modifier_concept_id = p_modifier.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.visit_occurrence` v 
                ON procedure.visit_occurrence_id = v.visit_occurrence_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_visit 
                ON v.visit_concept_id = p_visit.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` p_source_concept 
                ON procedure.procedure_source_concept_id = p_source_concept.concept_id"""

#dataset_procedure_df = pandas.read_gbq(
#    dataset_procedure_sql,
#    dialect="standard",
#    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
#    progress_bar_type="tqdm_notebook")

#dataset_procedure_df.head(5)
import pandas
import os

# This query represents dataset "pain_v_ppsp" for domain "condition" and was generated for All of Us Registered Tier Dataset v7
dataset_condition_sql = """
    SELECT
        c_occurrence.person_id,
        c_occurrence.condition_concept_id,
        c_standard_concept.concept_name as standard_concept_name,
        c_standard_concept.concept_code as standard_concept_code,
        c_standard_concept.vocabulary_id as standard_vocabulary,
        c_occurrence.condition_start_datetime,
        c_occurrence.condition_end_datetime,
        c_occurrence.condition_type_concept_id,
        c_type.concept_name as condition_type_concept_name,
        c_occurrence.stop_reason,
        c_occurrence.visit_occurrence_id,
        visit.concept_name as visit_occurrence_concept_name,
        c_occurrence.condition_source_value,
        c_occurrence.condition_source_concept_id,
        c_source_concept.concept_name as source_concept_name,
        c_source_concept.concept_code as source_concept_code,
        c_source_concept.vocabulary_id as source_vocabulary,
        c_occurrence.condition_status_source_value,
        c_occurrence.condition_status_concept_id,
        c_status.concept_name as condition_status_concept_name 
    FROM
        ( SELECT
            * 
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.condition_occurrence` c_occurrence 
        WHERE
            (
                condition_concept_id IN (SELECT
                    DISTINCT c.concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                JOIN
                    (SELECT
                        CAST(cr.id as string) AS id       
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                    WHERE
                        concept_id IN (4150125)       
                        AND full_text LIKE '%_rank1]%'      ) a 
                        ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                        OR c.path LIKE CONCAT('%.', a.id) 
                        OR c.path LIKE CONCAT(a.id, '.%') 
                        OR c.path = a.id) 
                WHERE
                    is_standard = 1 
                    AND is_selectable = 1)
            )  
            AND (
                c_occurrence.PERSON_ID IN (SELECT
                    distinct person_id  
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` cb_search_person  
                WHERE
                    cb_search_person.person_id IN (SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (4150125)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 1 
                                AND is_selectable = 1) 
                            AND is_standard = 1 )) criteria 
                    UNION
                    DISTINCT SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (4004516)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 1 
                                AND is_selectable = 1) 
                            AND is_standard = 1 )) criteria ) 
                    AND cb_search_person.person_id IN (SELECT
                        criteria.person_id 
                    FROM
                        (SELECT
                            DISTINCT person_id, entry_date, concept_id 
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                        WHERE
                            (concept_id IN(SELECT
                                DISTINCT c.concept_id 
                            FROM
                                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                            JOIN
                                (SELECT
                                    CAST(cr.id as string) AS id       
                                FROM
                                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                                WHERE
                                    concept_id IN (1585710, 43528895, 40192389, 1740639, 1586134, 1333342, 1741006, 1585855)       
                                    AND full_text LIKE '%_rank1]%'      ) a 
                                    ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                                    OR c.path LIKE CONCAT('%.', a.id) 
                                    OR c.path LIKE CONCAT(a.id, '.%') 
                                    OR c.path = a.id) 
                            WHERE
                                is_standard = 0 
                                AND is_selectable = 1) 
                            AND is_standard = 0 )) criteria ) )
                )
            ) c_occurrence 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_standard_concept 
                ON c_occurrence.condition_concept_id = c_standard_concept.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_type 
                ON c_occurrence.condition_type_concept_id = c_type.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.visit_occurrence` v 
                ON c_occurrence.visit_occurrence_id = v.visit_occurrence_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` visit 
                ON v.visit_concept_id = visit.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_source_concept 
                ON c_occurrence.condition_source_concept_id = c_source_concept.concept_id 
        LEFT JOIN
            `""" + os.environ["WORKSPACE_CDR"] + """.concept` c_status 
                ON c_occurrence.condition_status_concept_id = c_status.concept_id"""

dataset_condition_df = pandas.read_gbq(
    dataset_condition_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")
#%%
import pandas as pd
import hydra
from omegaconf import DictConfig
import pickle

# Generate mapping between data and the columns that we're using.
def get_column_mappers(config: DictConfig):
    columns = []
    column_to_type = {}

    codebook = pd.read_excel(os.path.join(os.path.dirname(__file__), config["data"]["codebook"]), sheet_name=None)
    for page_name, page in codebook.items():
        if any([ep in page_name for ep in config["data"]["exclude_pages"]]):
            continue

        print(f"Unique field types: {page[config['data']['field_type']].unique()}")

        for _, field in page.iterrows():
            # Skip fields that are "text" without any validation
            # Skip descriptive fields 
            if field[config["data"]["field_type"]] == "descriptive" or field[config["data"]["field_type"]] == "text":
                continue
            choices = None
            if isinstance(field[config["data"]["choices"]], str) and "|" in field[config["data"]["choices"]]:
                if field[config["data"]["field_type"]] == "slider":
                    choices = None # Just a number
                elif field[config["data"]["field_type"]] == "checkbox" or field[config["data"]["field_type"]] == "radio" or field[config["data"]["field_type"]] == "dropdown":
                    choices = [x.split("_")[-1] for x in field[config["data"]["choices"]].split("|")]
                    if len(choices) == 2:
                        choices = None # Binary
                    elif "old" in field[config["data"]["field_label"]].lower() or "new" in field[config["data"]["field_label"]].lower():
                        choices = None # Age groups, ordinal
                    elif len(choices) > config["data"]["max_categories"]:
                        continue
                else:
                    choices = None
            f_name = field[config["data"]["field_label"]]
            if choices is not None:
                columns.extend([f"{f_name}_{choice}" for choice in choices])
                column_to_type[f_name] = ("one_hot", choices)
            elif not pd.isna(f_name):
                columns.append(f_name)
                column_to_type[f_name] = ("scalar", None)
    return columns, column_to_type
            
#%%

from tqdm import tqdm

@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf"), config_name="config")
def main(config: DictConfig):
    # Get the list of patients
    patients = dataset_person_df["person_id"].unique()
    # Test if there's a condition matching the criteria for each patient
    columns, column_to_type = get_column_mappers(config)
    conditions = config["data"]["y_labels"]
    results = []
    for patient in tqdm(patients):
        patient_conditions = dataset_condition_df[dataset_condition_df["person_id"] == patient]["source_concept_name"].unique()
        # Serialize all the questions into a dict
        questions = {column: 0 for column in columns}
        for index, row in dataset_survey_df[dataset_survey_df["person_id"] == patient].iterrows():
            if row["question"] not in column_to_type:
                continue
            if column_to_type[row["question"]][0]  == "one_hot":
                # Find column name, then one hot name
                # Don't split here
                cand_names = [x for x in columns if row["question"] in x]
                one_hot_name = [x for x in filter(lambda x: row["answer"] in x.split(",")[1], cand_names)]
                if len(one_hot_name) == 0:
                    continue
                one_hot_name = one_hot_name[0]
                questions[one_hot_name] = 1
            else:
                try:
                    questions[row["question"]] = int(re.match(r"\d+", row["answer"]).group(0))
                except (ValueError, AttributeError):
                    questions[row["question"]] = 1 if "yes" in row["answer"].lower() else 0
        for condition in conditions:
            # Serialize the patient data
            patient_conditions = [x for x in patient_conditions if not pd.isna(x)]
            questions[condition] = 1 if any(["chronic" in x.lower() for x in patient_conditions]) else 0

        results.append(questions)

    # Save the data
    df = pd.DataFrame(results)
    df.to_csv(config["path"]["source"])

if __name__ == "__main__":
    main()
