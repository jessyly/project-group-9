########################## CREATE INDICES ##########################

CREATE INDEX idx_pbs_drugs_pbs_code
on pbs_drugs(pbs_code);

CREATE INDEX idx_pbs_drugs_file_date
on pbs_drugs(file_date);

CREATE INDEX idx_pbs_drugs_pbs_cd_file_date
on pbs_link(pbs_code, file_date);

CREATE INDEX idx_pbs_link_ind_id
on pbs_link(indication_id);

CREATE INDEX idx_pbs_link_item_cd
on pbs_link(pbs_item_code);

CREATE INDEX idx_pbs_link_file_date
on pbs_link(file_date);

CREATE INDEX idx_pbs_link_pbs_cd_file_date
on pbs_link(pbs_item_code, file_date);

CREATE INDEX idx_pbs_link_ind_id_file_date
on pbs_link(indication_id, file_date);

CREATE INDEX idx_pbs_res_ind_id
on pbs_restrictions(indication_id);

CREATE INDEX idx_pbs_res_file_date
on pbs_restrictions(file_date);

CREATE INDEX idx_pbs_res_ind_id_file_date
on pbs_restrictions(indication_id, file_date);

########################## CREATE PBS CODE TO DISEASE CONDITION MAPPING TABLE ##########################

DROP TABLE IF EXISTS pbs_code_to_condition_mapping;
CREATE TABLE pbs_code_to_condition_mapping AS
SELECT   nhs_dispensed_code
	   , streamlined_approval_code
       , drug_name
	   , restriction_flag
	   , disease_condition 
	   , file_date
	   , restriction_indication_full_text
FROM
(
	SELECT   nhs_dispensed_code
		   , streamlined_approval_code
	       , drug_name
		   , restriction_flag
		   , disease_condition 
		   , file_date
		   , restriction_indication_full_text
		   , ROW_NUMBER() OVER (PARTITION BY nhs_dispensed_code, streamlined_approval_code, restriction_flag ORDER BY LENGTH(disease_condition) ASC) rn
	FROM
	(
		SELECT   nhs_dispensed_code 
			   , streamlined_approval_code 
		       , drug_name
			   , restriction_flag 
			   , disease_condition 
			   , file_date
			   , restriction_indication_full_text
			   , ROW_NUMBER() OVER (PARTITION BY nhs_dispensed_code, streamlined_approval_code, restriction_flag, SOUNDEX(disease_condition) ORDER BY file_date ASC) rn1
			   , ROW_NUMBER() OVER (PARTITION BY nhs_dispensed_code, streamlined_approval_code, restriction_flag, REPLACE(restriction_indication_full_text, ' ', '') ORDER BY LENGTH(disease_condition) ASC, file_date ASC) rn2 # for cases where different number of spaces in restriction text to extract disease
		FROM
		(
			SELECT DISTINCT   d.pbs_code nhs_dispensed_code
							, r.indication_id streamlined_approval_code
						    , d.drug_mp_name drug_name
							, d.restriction_flag
							, TRIM(BOTH '"' FROM LEFT(r.restriction_indication_text, 
									CASE WHEN		(instr(r.restriction_indication_text, 'Treatment Phase:') = 0) 
												AND (instr(r.restriction_indication_text, 'Treatment Criteria:') = 0) 
												AND (instr(r.restriction_indication_text, 'Population criteria:') = 0)
												AND (instr(r.restriction_indication_text, 'Clinical criteria:') = 0)
												AND (instr(r.restriction_indication_text, 'The Clinical criteria is:') = 0)
												AND (instr(r.restriction_indication_text, 'The Treatment criteria is:') = 0)
												AND (instr(r.restriction_indication_text, 'The Population criteria is:') = 0)
										THEN 
											(
												CASE WHEN instr(r.restriction_indication_text, '  ') != 0
													 THEN instr(r.restriction_indication_text, '  ')
													 WHEN instr(r.restriction_indication_text, '   ') != 0
													 THEN instr(r.restriction_indication_text, '   ')
													 WHEN (instr(r.restriction_indication_text, '.') != 0) 
													 THEN instr(r.restriction_indication_text, '.')
													 ELSE LENGTH(r.restriction_indication_text) END
											)
									ELSE
									LEAST(IFNULL(NULLIF(instr(r.restriction_indication_text, 'Treatment Phase:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'Treatment criteria:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'Population criteria:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'Clinical criteria:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'The Clinical criteria is:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'The Treatment criteria is:'),0),1000000), IFNULL(NULLIF(instr(r.restriction_indication_text, 'The Population criteria is:'),0),1000000)) - 2 END)) disease_condition
							, d.file_date
							, TRIM(BOTH '"' FROM r.restriction_indication_text) restriction_indication_full_text
			FROM predictpatientoutcomes.pbs_drugs d
			INNER JOIN predictpatientoutcomes.pbs_link l
			ON d.pbs_code = l.pbs_item_code AND d.file_date = l.file_date
			INNER JOIN predictpatientoutcomes.pbs_restrictions r
			ON l.indication_id = r.indication_id AND l.file_date = r.file_date
		) A
	) B WHERE rn1 = 1 AND rn2 = 1
) C
WHERE rn = 1;