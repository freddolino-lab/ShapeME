CREATE TABLE users (
    uid integer primary key,
    first text,
    last text,
    email text UNIQUE,
    password text,
    institution text
);

CREATE TABLE jobs (
    id integer primary key,
    args text,
    version text,
    uid integer,
    FOREIGN KEY (uid)
        REFERENCES users (uid)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

PRAGMA foreign_keys = ON;

INSERT INTO users (first, last, email, password) VALUES ("Test", "User", "test_email@fake.com", "111111");
INSERT INTO jobs (args, version, uid) VALUES ("", "0.1.0", 1);
INSERT INTO jobs (args, version, uid) VALUES ("", "0.1.0", 2);
