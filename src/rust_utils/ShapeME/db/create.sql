CREATE TABLE users (
    uid integer primary key,
    first text,
    last text,
    email text,
    password text,
    institution text
);

CREATE TABLE jobs (
    id text primary key,
    args text,
    version text,
    uid integer,
    FOREIGN KEY (uid)
        REFERENCES users (uid)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);
