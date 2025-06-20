create table metrics
(
    id           int          not null
        primary key,
    metric_code  varchar(100) not null,
    display_name varchar(255) not null,
    constraint metric_code
        unique (metric_code)
);

create table financial_data
(
    id             int auto_increment
        primary key,
    company_ticker varchar(10)                          not null,
    metric_id      int                                  not null,
    fiscal_year    int                                  not null,
    fiscal_period  enum ('Q1', 'Q2', 'Q3', 'Q4', 'FY')  not null,
    value          bigint                               null,
    currency       varchar(3) default 'USD'             null,
    filing_date    date                                 null,
    extracted_at   timestamp  default CURRENT_TIMESTAMP null,
    constraint idx_unique_metric
        unique (company_ticker, metric_id, fiscal_year, fiscal_period),
    constraint financial_data_ibfk_1
        foreign key (metric_id) references metrics (id)
);

create index metric_id
    on financial_data (metric_id);

