CREATE TABLE IF NOT EXISTS public.invoices
(
    id integer,
    date date,
    voucher_name character varying COLLATE pg_catalog."default",
    branch character varying COLLATE pg_catalog."default",
    currency character varying COLLATE pg_catalog."default",
    party character varying COLLATE pg_catalog."default",
    quantity numeric,
    gross_amt numeric,
    discount numeric,
    gross_minus_discount numeric,
    net_amount numeric,
    deductions numeric,
    total_value numeric
)

CREATE TABLE IF NOT EXISTS public.reconciliation
(
    id integer NOT NULL,
    po_id character varying COLLATE pg_catalog."default",
    invoices integer,
    delivery_challan integer,
    full_matches integer,
    partial_matches integer,
    rejections integer,
    rejection_reason character varying COLLATE pg_catalog."default",
    tax_discrepancy integer,
    currency_mismatch character varying COLLATE pg_catalog."default",
    item_mismatch numeric,
    date date,
    vendor character varying COLLATE pg_catalog."default",
    CONSTRAINT reconciliation_pkey PRIMARY KEY (id)
)