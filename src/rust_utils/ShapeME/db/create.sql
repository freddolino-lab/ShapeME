CREATE TABLE users (
    uid integer primary key,
    first text,
    last text,
    email text UNIQUE,
    password text,
    institution text
);

CREATE TABLE jobs (
    id text NOT NULL UNIQUE,
    name text,
    args text,
    version text,
    uid integer,
    FOREIGN KEY (uid)
        REFERENCES users (uid)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

PRAGMA foreign_keys = ON;

INSERT INTO users (first, last, email, password) VALUES ("Test", "User", "test_email@fake.com", "855b797db4d3cfae2fb40101ae299d75fae1ce6f41b2f067a4f1561409f49801");
INSERT INTO jobs (id, name, args, version, uid) VALUES ("0O3YSDFJK", "testname", "args", "0.1.0", 1);
INSERT INTO jobs (id, name, args, version, uid) VALUES ("0O3YSDFJK", "shouldn't exist (non-unique id)", "args", "0.1.0", 1);
INSERT INTO jobs (id, name, args, version, uid) VALUES ("SDFSKJDLFKJS", "shouldn't exist (no uid in users)", "args", "0.1.0", 2);
