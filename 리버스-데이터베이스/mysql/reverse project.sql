create table 성적 (
	학번 varchar(20) not null,
    과목코드 varchar(20) not null,
    성적 varchar(20),
    primary key (학번, 과목코드),
	foreign key (학번) references 학생(학번),
    foreign key (과목코드) references 과목(과목코드)
);