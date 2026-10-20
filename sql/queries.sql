-- Backlog queries shown in the app. Each block starts with "-- name: <key>"
-- and a "-- title:" line; db.py splits the file on those markers.
-- Rates use AVG(scam * 1.0) so integer division never truncates them.

-- name: overview
-- title: Backlog at a glance
SELECT COUNT(*)                                          AS postings,
       SUM(scam)                                         AS confirmed_scams,
       AVG(scam * 1.0)                                   AS scam_rate,
       SUM(CASE WHEN salary_min IS NULL THEN 1 ELSE 0 END) AS without_salary
FROM posting;

-- name: profile_by_outcome
-- title: Average listing traits, legitimate vs scam
SELECT CASE WHEN scam = 1 THEN 'scam' ELSE 'legitimate' END AS outcome,
       COUNT(*)                                  AS postings,
       AVG(required_experience_years)            AS experience_years,
       AVG(skills_listed * 1.0)                  AS skills_listed,
       AVG(vague_word_count * 1.0)               AS vague_words,
       AVG(exclamation_marks * 1.0)              AS exclamation_marks,
       AVG(allcaps_ratio)                        AS allcaps_ratio,
       AVG(employer_response_delay_h)            AS response_delay_h,
       AVG(company_age_months * 1.0)             AS company_age_months,
       AVG(company_size * 1.0)                   AS company_size,
       AVG(salary_max * 1.0 / salary_min)        AS salary_width
FROM posting
GROUP BY scam
ORDER BY scam;

-- name: rate_by_category
-- title: Scam rate per job category
SELECT c.name                 AS category,
       COUNT(*)               AS postings,
       SUM(p.scam)            AS scams,
       AVG(p.scam * 1.0)      AS scam_rate
FROM posting p
JOIN category c ON c.category_id = p.category_id
GROUP BY c.name
ORDER BY scam_rate DESC;

-- name: rate_by_salary_width
-- title: Scam rate by how wide the advertised pay range is
SELECT CASE
           WHEN salary_min IS NULL                 THEN '0 no range published'
           WHEN salary_max * 1.0 / salary_min <= 1.2 THEN '1 narrow (max <= 1.2 x min)'
           WHEN salary_max * 1.0 / salary_min <= 1.4 THEN '2 medium (1.2 - 1.4 x)'
           ELSE                                         '3 wide (> 1.4 x min)'
       END                     AS salary_range,
       COUNT(*)                AS postings,
       SUM(scam)               AS scams,
       AVG(scam * 1.0)         AS scam_rate
FROM posting
GROUP BY 1
ORDER BY 1;

-- name: rate_by_posting_time
-- title: Scam rate by time of day the listing went live
SELECT CASE
           WHEN posting_hour < 6  THEN '1 night (00-05)'
           WHEN posting_hour < 12 THEN '2 morning (06-11)'
           WHEN posting_hour < 18 THEN '3 afternoon (12-17)'
           ELSE                        '4 evening (18-23)'
       END                     AS time_of_day,
       COUNT(*)                AS postings,
       SUM(scam)               AS scams,
       AVG(scam * 1.0)         AS scam_rate
FROM posting
GROUP BY 1
ORDER BY 1;

-- name: rate_by_experience
-- title: Scam rate by years of experience demanded
SELECT CASE
           WHEN required_experience_years < 2 THEN '1 under 2 years'
           WHEN required_experience_years < 5 THEN '2 2 to 5 years'
           ELSE                                    '3 5 years or more'
       END                     AS experience,
       COUNT(*)                AS postings,
       SUM(scam)               AS scams,
       AVG(scam * 1.0)         AS scam_rate
FROM posting
GROUP BY 1
ORDER BY 1;

-- name: rate_by_flag
-- title: Scam rate when a yes/no flag is set
SELECT 'asks for trial fee or payment' AS flag,
       SUM(mentions_trial_or_payment)  AS postings_with_flag,
       AVG(CASE WHEN mentions_trial_or_payment = 1 THEN scam * 1.0 END) AS scam_rate_with,
       AVG(CASE WHEN mentions_trial_or_payment = 0 THEN scam * 1.0 END) AS scam_rate_without
FROM posting
UNION ALL
SELECT 'gmail / hotmail contact', SUM(free_mail_contact),
       AVG(CASE WHEN free_mail_contact = 1 THEN scam * 1.0 END),
       AVG(CASE WHEN free_mail_contact = 0 THEN scam * 1.0 END)
FROM posting
UNION ALL
SELECT 'remote job', SUM(is_remote),
       AVG(CASE WHEN is_remote = 1 THEN scam * 1.0 END),
       AVG(CASE WHEN is_remote = 0 THEN scam * 1.0 END)
FROM posting;

-- name: largest_claimed_scam_company
-- title: Largest company size claimed by a confirmed scam vs legitimate posting
SELECT CASE WHEN scam = 1 THEN 'scam' ELSE 'legitimate' END AS outcome,
       MIN(company_size) AS smallest,
       MAX(company_size) AS largest
FROM posting
GROUP BY scam
ORDER BY scam;

-- name: all_postings
-- title: All reviewed postings (used by the model)
SELECT p.posting_id, p.salary_min, p.salary_max, p.required_experience_years,
       p.skills_listed, p.vague_word_count, p.exclamation_marks, p.allcaps_ratio,
       p.posting_hour, p.employer_response_delay_h, p.is_remote,
       p.free_mail_contact, p.company_age_months, p.company_size,
       c.name AS category, p.mentions_trial_or_payment, p.scam
FROM posting p
JOIN category c ON c.category_id = p.category_id
ORDER BY p.posting_id;
