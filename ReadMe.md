# Hospital Inpatient Discharges (SPARCS) - Big Data Project

We created a Ridge Regression model for Total Charges in the SPARCS dataset. 

## References

- [NY Health Data: Hospital Inpatient Discharges (SPARCS)](https://health.data.ny.gov/stories/s/wvua-rr23)
- [About the Dataset](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/sf4k-39ay/about_data)
- [Project Notes (Google Doc)](https://docs.google.com/document/d/13sy38fD6CC0Nsm3qKmerP2H5CCCc4WqqCawlnaGR0Y0/edit?tab=t.0)
- [ChatGPT Session](https://chatgpt.com/c/68f6ff37-9e90-832f-893c-5388df69f950)

---

## Workflow & Commands

### 1. Transfer Files to EC2

```sh
scp -i localDocuments/sem4/ds644/midTermProject/DS644.pem /Users/sv.xxt/localDocuments/sem4/ds644/midTermProject/midtermBigData/RidgeRegressionCVMR_Intercept.java ubuntu@ec2-54-82-28-144.compute-1.amazonaws.com:~/
scp -i localDocuments/sem4/ds644/midTermProject/DS644.pem /Users/sv.xxt/localDocuments/sem4/ds644/midTermProject/midtermBigData/train.csv ubuntu@ec2-54-82-28-144.compute-1.amazonaws.com:~/
```

### 2. Prepare Directories on EC2

```sh
mkdir -p ~/ridge-mr/{src,build,lib,logs}
mv ~/RidgeRegressionCVMR_Intercept.java ~/ridge-mr/src/
```

### 3. Download Dependencies

```sh
cd ~/ridge-mr/lib
curl -L -o commons-math3-3.6.1.jar \
  https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar
```

### 4. Compile Java Code

```sh
cd ~/ridge-mr
javac -source 1.8 -target 1.8 \
  -cp "$(hadoop classpath):./lib/commons-math3-3.6.1.jar" \
  -d build src/RidgeRegressionCVMR_Intercept.java
```

### 5. Create JAR File

```sh
echo "Main-Class: RidgeRegressionCVMR_Intercept" > manifest.txt
jar cfm ridge-mr.jar manifest.txt -C build .
```

### 6. Upload Data to HDFS

```sh
hdfs dfs -mkdir -p /user/$USER
hdfs dfs -put -f ~/train.csv /user/$USER/train.csv
hdfs dfs -mkdir -p /user/$USER/ridge_work
```

### 7. Set Hadoop Classpath

```sh
export HADOOP_CLASSPATH="$HADOOP_CLASSPATH:$(pwd)/lib/commons-math3-3.6.1.jar"
```

### 8. Run Ridge Regression on Hadoop

```sh
hadoop jar ridge-mr.jar \
  -libjars "$(pwd)/lib/commons-math3-3.6.1.jar" \
  /user/$USER/train.csv /user/$USER/ridge_work \
  132 1.0 5 , run1 true true
```

### 9. View Results

```sh
for f in {0..4}; do
  printf "Fold %d: " "$f"
  hdfs dfs -cat /user/$USER/ridge_work/eval_run1_fold_${f}/part-r-00000
done
```

#### Calculate Mean R² Across Folds

```sh
for f in {0..4}; do
  hdfs dfs -cat /user/$USER/ridge_work/eval_run1_fold_${f}/part-r-00000
done | awk -F 'r2=' '
  { split($2,a,"[, ]"); if(a[1]!="") {n++; s+=a[1]; printf("r2[%d]=%s\n", n, a[1])}}
  END { if(n>0) printf("Mean R^2 over %d folds = %.6f\n", n, s/n); else print "No R^2 found"; }'
```

---

## Notes

- Make sure you have the correct permissions and your Hadoop cluster is running.
- Adjust file paths and user names as needed for your environment.
- The workflow assumes you have Java, Hadoop, and all dependencies installed on your EC2 instance.

---
