-- ================================================
-- CSP PROJECT - SQL QUERIES
-- Database: customer_support
-- Table: customer_support_tickets
-- ================================================

-- ================================================
-- 1. CREATE TABLE
-- ================================================

CREATE TABLE IF NOT EXISTS customer_support_tickets (
    ticket_id                   INT PRIMARY KEY,
    customer_name               VARCHAR(100),
    customer_email              VARCHAR(100),
    customer_age                INT,
    customer_gender             VARCHAR(20),
    product_purchased           VARCHAR(100),
    date_of_purchase            DATE,
    ticket_type                 VARCHAR(50),
    ticket_subject              VARCHAR(100),
    ticket_description          TEXT,
    ticket_status               VARCHAR(50),
    resolution                  TEXT,
    ticket_priority             VARCHAR(20),
    ticket_channel              VARCHAR(50),
    first_response_time         DATETIME,
    time_to_resolution          DATETIME,
    customer_satisfaction_rating FLOAT
);

-- ================================================
-- 2. BASIC EXPLORATION
-- ================================================

SELECT COUNT(*) AS total_tickets
FROM customer_support_tickets;

SELECT COUNT(*) AS rated_tickets
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL;

SELECT COUNT(*) AS unrated_tickets
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NULL;

-- ================================================
-- 3. CUSTOMER SATISFACTION ANALYSIS
-- ================================================

SELECT
    customer_satisfaction_rating AS rating,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY customer_satisfaction_rating
ORDER BY rating;

SELECT
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction,
    MIN(customer_satisfaction_rating)            AS min_satisfaction,
    MAX(customer_satisfaction_rating)            AS max_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL;

-- ================================================
-- 4. TICKET TRENDS ANALYSIS
-- ================================================

SELECT
    ticket_type,
    COUNT(*) AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY ticket_type
ORDER BY total_tickets DESC;

SELECT
    ticket_priority,
    COUNT(*) AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY ticket_priority
ORDER BY avg_satisfaction DESC;

SELECT
    ticket_channel,
    COUNT(*) AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY ticket_channel
ORDER BY avg_satisfaction DESC;

SELECT
    ticket_subject,
    COUNT(*) AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY ticket_subject
ORDER BY total_tickets DESC
LIMIT 10;

-- ================================================
-- 5. CUSTOMER SEGMENTATION
-- ================================================

SELECT
    customer_gender,
    COUNT(*)                                     AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2)  AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY customer_gender
ORDER BY avg_satisfaction DESC;

SELECT
    CASE
        WHEN customer_age BETWEEN 18 AND 25 THEN '18-25'
        WHEN customer_age BETWEEN 26 AND 35 THEN '26-35'
        WHEN customer_age BETWEEN 36 AND 50 THEN '36-50'
        WHEN customer_age BETWEEN 51 AND 65 THEN '51-65'
        ELSE '65+'
    END AS age_group,
    COUNT(*)                                    AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY age_group
ORDER BY age_group;

SELECT
    product_purchased,
    COUNT(*)                                    AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY product_purchased
ORDER BY avg_satisfaction DESC
LIMIT 10;

-- ================================================
-- 6. TICKET STATUS ANALYSIS
-- ================================================

SELECT
    ticket_status,
    COUNT(*) AS total,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM customer_support_tickets
GROUP BY ticket_status;

SELECT
    ticket_status,
    ticket_type,
    COUNT(*) AS total
FROM customer_support_tickets
GROUP BY ticket_status, ticket_type
ORDER BY ticket_status, total DESC;

-- ================================================
-- 7. HIGH & LOW SATISFACTION TICKETS
-- ================================================

SELECT
    ticket_type,
    ticket_priority,
    ticket_channel,
    customer_satisfaction_rating
FROM customer_support_tickets
WHERE customer_satisfaction_rating = 5
ORDER BY ticket_type;

SELECT
    ticket_type,
    ticket_priority,
    ticket_channel,
    customer_satisfaction_rating
FROM customer_support_tickets
WHERE customer_satisfaction_rating = 1
ORDER BY ticket_type;

-- ================================================
-- 8. MONTHLY TICKET TRENDS
-- ================================================

SELECT
    YEAR(date_of_purchase)  AS year,
    MONTH(date_of_purchase) AS month,
    COUNT(*)                AS total_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2) AS avg_satisfaction
FROM customer_support_tickets
WHERE customer_satisfaction_rating IS NOT NULL
GROUP BY year, month
ORDER BY year, month;

-- ================================================
-- 9. TOP 5 MOST COMMON ISSUES BY CHANNEL
-- ================================================

SELECT
    ticket_channel,
    ticket_subject,
    COUNT(*) AS total
FROM customer_support_tickets
GROUP BY ticket_channel, ticket_subject
ORDER BY ticket_channel, total DESC
LIMIT 20;

-- ================================================
-- 10. OVERALL SUMMARY REPORT
-- ================================================

SELECT
    COUNT(*)                                       AS total_tickets,
    COUNT(customer_satisfaction_rating)            AS rated_tickets,
    ROUND(AVG(customer_satisfaction_rating), 2)    AS avg_satisfaction,
    SUM(CASE WHEN ticket_status = 'Closed'
             THEN 1 ELSE 0 END)                    AS closed_tickets,
    SUM(CASE WHEN ticket_status = 'Open'
             THEN 1 ELSE 0 END)                    AS open_tickets,
    SUM(CASE WHEN ticket_status = 'Pending Customer Response'
             THEN 1 ELSE 0 END)                    AS pending_tickets
FROM customer_support_tickets;
