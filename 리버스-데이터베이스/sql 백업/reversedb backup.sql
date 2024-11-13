CREATE DATABASE  IF NOT EXISTS `학생` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `학생`;
-- MySQL dump 10.13  Distrib 8.0.38, for Win64 (x86_64)
--
-- Host: localhost    Database: 학생
-- ------------------------------------------------------
-- Server version	8.0.39

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `과목`
--

DROP TABLE IF EXISTS `과목`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `과목` (
  `과목코드` varchar(20) NOT NULL,
  `과목명` varchar(20) NOT NULL,
  `이수구분` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`과목코드`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `과목`
--

LOCK TABLES `과목` WRITE;
/*!40000 ALTER TABLE `과목` DISABLE KEYS */;
INSERT INTO `과목` VALUES ('00024','채플','엘림교양'),('10050','모바일프로그래밍','전공선택'),('10054','컴퓨터개론','전공선택'),('11005','이산수학','전공선택'),('11013','컴퓨터구조','전공선택'),('11014','시스템프로그래밍','전공선택'),('11023','데이터베이스','전공선택'),('11050','데이터통신','전공선택'),('11071','논리회로','전공선택'),('11072','소프트웨어공학','전공필수'),('11098','자료구조','전공필수'),('11099','객체지향프로그래밍1','전공선택'),('11102','운영체제','전공선택'),('11109','객체지향프로그래밍2','전공필수'),('11111','컴퓨터네트워크','전공선택'),('11134','확률과통계','전공선택'),('11614','자바프로그래밍1','전공필수'),('11616','알고리즘분석및실습','전공선택'),('11617','자바프로그래밍2','전공필수'),('12970','[캡스톤]캠스톤디자인1','전공선택'),('13230','[졸업작품][캠스톤]캠스톤디자인2','전공선택'),('13361','창업실습I','전공선택'),('13863','정보보안관리','전공선택'),('13864','정보보안기술','전공선택'),('13867','프로그래밍언어1(C언어)','전공선택'),('13868','프로그래밍언어2(C언어)','전공필수'),('13903','세계문학읽기','스템교양'),('14653','인간발달과융합적사고','스템교양'),('14823','데이터베이스구축실습','전공선택'),('16905','빅데이터컴퓨팅','전공선택'),('22227','AI개론','전공선택'),('22228','데이터과학수학','전공선택'),('22230','창업실습2','전공선택'),('22231','빅데이터분석실습','전공선택'),('22232','가상화클라우드컴퓨팅','전공선택'),('22233','산학프로젝트1','전공선택'),('22234','머신러닝과데이터분석','전공선택'),('22235','SW안전실습','전공선택'),('22236','산학프로젝트2','전공선택');
/*!40000 ALTER TABLE `과목` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `성적`
--

DROP TABLE IF EXISTS `성적`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `성적` (
  `학번` varchar(20) NOT NULL,
  `과목코드` varchar(20) NOT NULL,
  `성적` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`학번`,`과목코드`),
  KEY `과목코드` (`과목코드`),
  CONSTRAINT `성적_ibfk_1` FOREIGN KEY (`학번`) REFERENCES `학생` (`학번`),
  CONSTRAINT `성적_ibfk_2` FOREIGN KEY (`과목코드`) REFERENCES `과목` (`과목코드`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `성적`
--

LOCK TABLES `성적` WRITE;
/*!40000 ALTER TABLE `성적` DISABLE KEYS */;
INSERT INTO `성적` VALUES ('21101862','10054','A'),('21101862','11013','A'),('21101862','11071','A+'),('21101862','11072','C+'),('21101862','11098','A+'),('21101862','11099','A'),('21101862','13867','B'),('21101862','13868','B+');
/*!40000 ALTER TABLE `성적` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `시간표`
--

DROP TABLE IF EXISTS `시간표`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `시간표` (
  `시간표_id` varchar(20) NOT NULL,
  `과목명` varchar(20) DEFAULT NULL,
  `과목코드` varchar(20) DEFAULT NULL,
  `요일` varchar(20) DEFAULT NULL,
  `시작시간` time DEFAULT NULL,
  `종료시간` time DEFAULT NULL,
  `학번` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`시간표_id`),
  KEY `과목코드` (`과목코드`),
  KEY `학번` (`학번`),
  CONSTRAINT `시간표_ibfk_1` FOREIGN KEY (`과목코드`) REFERENCES `과목` (`과목코드`),
  CONSTRAINT `시간표_ibfk_2` FOREIGN KEY (`학번`) REFERENCES `학생` (`학번`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `시간표`
--

LOCK TABLES `시간표` WRITE;
/*!40000 ALTER TABLE `시간표` DISABLE KEYS */;
INSERT INTO `시간표` VALUES ('2024_2_1','알고리즘분석및실습','11616','월','09:00:00','13:00:00','21101862'),('2024_2_2','확률과통계','11134','월','14:00:00','17:00:00','21101862'),('2024_2_3','채플','00024','화','14:00:00','15:00:00','21101862'),('2024_2_4','객체지향프로그래밍2','11109','수','09:00:00','13:00:00','21101862'),('2024_2_5','데이터베이스','11023','수','14:00:00','17:00:00','21101862'),('2024_2_6','인간발달과융합적사고','14653','목','10:00:00','13:00:00','21101862'),('2024_2_7','세계문학읽기','13903','목','14:00:00','17:00:00','21101862');
/*!40000 ALTER TABLE `시간표` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `학생`
--

DROP TABLE IF EXISTS `학생`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `학생` (
  `학번` varchar(20) NOT NULL,
  `학년` int DEFAULT NULL,
  `이름` varchar(20) NOT NULL,
  `생년월일` date DEFAULT NULL,
  `전화번호` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`학번`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `학생`
--

LOCK TABLES `학생` WRITE;
/*!40000 ALTER TABLE `학생` DISABLE KEYS */;
INSERT INTO `학생` VALUES ('21101862',2,'천민기','2002-08-07','010-2367-1347');
/*!40000 ALTER TABLE `학생` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database '학생'
--

--
-- Dumping routines for database '학생'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-10-10 19:45:10
