
  CREATE TABLE "CREDIT" 
   (	"ID" NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY MINVALUE 1 MAXVALUE 9999999999999999999999999999 INCREMENT BY 1 START WITH 1 CACHE 20 NOORDER  NOCYCLE  NOKEEP  NOSCALE  NOT NULL ENABLE, 
	"LOAN" NUMBER, 
	"MORTDUE" NUMBER, 
	"VALUE" NUMBER, 
	"REASON" VARCHAR2(25), 
	"JOB" VARCHAR2(25), 
	"YOJ" NUMBER, 
	"DEROG" NUMBER, 
	"DELINQ" NUMBER, 
	"CLAGE" NUMBER, 
	"NINQ" NUMBER, 
	"CLNO" NUMBER, 
	"DEBTINC" NUMBER, 
	"PROBA" NUMBER, 
	"SCORE" NUMBER, 
	"CREATEDDATE" TIMESTAMP (6) DEFAULT CURRENT_TIMESTAMP, 
	"MODIFIEDDATE" TIMESTAMP (6) DEFAULT CURRENT_TIMESTAMP, 
	"STATE" VARCHAR2(255) DEFAULT 'PENDING', 
	 PRIMARY KEY ("ID")
  USING INDEX  ENABLE
   ) ;

  CREATE OR REPLACE EDITIONABLE TRIGGER "TRG_CREDIT_MODDATE" 
BEFORE UPDATE ON CREDIT
FOR EACH ROW
BEGIN
    :NEW.ModifiedDate := CURRENT_TIMESTAMP;
END;
/
ALTER TRIGGER "TRG_CREDIT_MODDATE" ENABLE;