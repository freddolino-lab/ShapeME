PRAGMA key = 'dont look at me';
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    uid integer primary key,
    first text,
    last text,
    email text UNIQUE,
    password text,
    salt text,
    institution text
);

CREATE TABLE jobs (
    id text NOT NULL UNIQUE,
    name text,
    args text,
    version text,
    path text,
    fa_path text,
    score_path text,
    uid integer,
    FOREIGN KEY (uid)
        REFERENCES users (uid)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

INSERT INTO users (first, last, email, password) VALUES ("Test", "User", "test_email@fake.com", "855b797db4d3cfae2fb40101ae299d75fae1ce6f41b2f067a4f1561409f49801");
INSERT INTO jobs (id, name, args, version, path, fa_path, score_path, uid) VALUES ("0O3YSDFJK", "testname", "args", "0.1.0", "don't ", "worry", "boutit", 1);
INSERT INTO jobs (id, name, args, version, path, fa_path, score_path, uid) VALUES ("0O3YSDFJK", "shouldn't exist (non-unique id)", "args", "0.1.0", "no", "path", "here", 1);
INSERT INTO jobs (id, name, args, version, path, fa_path, score_path, uid) VALUES ("SDFSKJDLFKJS", "shouldn't exist (no uid in users)", "args", "0.1.0", "care", "gonna", "not", 2);
