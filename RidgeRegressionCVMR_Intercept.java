// Hadoop Ridge Regression with K-fold CV and optional intercept. // Overview
// Input rows: CSV "y,f1,f2,...,fp" (NO intercept column).          // Input format
// Prints per-fold R^2 and mean R^2 to stdout.                       // Output metric
// Hadoop 2.6.5 / Java 8 compatible.                                 // Compatibility note

import org.apache.hadoop.conf.Configuration;                         // Hadoop config
import org.apache.hadoop.conf.Configured;                            // Tool base class helper
import org.apache.hadoop.fs.*;                                       // HDFS API
import org.apache.hadoop.io.*;                                       // Hadoop Writables
import org.apache.hadoop.mapreduce.*;                                // MapReduce core
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;        // Input path config
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;      // Output path config
import org.apache.hadoop.util.Tool;                                  // Tool interface
import org.apache.hadoop.util.ToolRunner;                            // Tool runner helper

import java.io.*;                                                    // IO utilities
import java.nio.charset.StandardCharsets;                            // Charset for reading files
import java.util.*;                                                  // Collections utilities
import java.util.Locale;                                             // Locale for printf formatting

import org.apache.commons.math3.linear.*;                            // Linear algebra (Cholesky/QR/etc)

public class RidgeRegressionCVMR_Intercept extends Configured implements Tool { // Main class implementing Tool

    // Writable for packed upper-triangular Gram (X^T X) and cross term (X^T y) // Custom value type
    public static class GramWritable implements Writable {            // Holds sufficient stats
        private int pEff;            // effective dimension (pRaw + 1 if intercept, else pRaw) // Fields
        private double[] gramUpper;  // upper-tri packed length = pEff*(pEff+1)/2              // Packed Gram
        private double[] xty;        // length = pEff                                           // Cross-term
        private long n;              // row count (diagnostic)                                  // Count rows

        public GramWritable() {}                                       // Hadoop requires no-arg ctor
        public GramWritable(int pEff) {                                // Construct with dimension
            this.pEff = pEff;                                          // Store dimension
            this.gramUpper = new double[pEff * (pEff + 1) / 2];        // Allocate packed Gram
            this.xty = new double[pEff];                               // Allocate cross vector
            this.n = 0L;                                               // Init count
        }
        public int dim() { return pEff; }                              // Accessor for dimension

        // Accumulate one (x, y) into Gram and b                                       
        public void add(double[] x, double y) {                        // Add a sample
            int idx = 0;                                               // Packed index cursor
            for (int i = 0; i < pEff; i++) {                           // Loop rows
                double xi = x[i];                                      // Cache x_i
                for (int j = i; j < pEff; j++) {                       // Loop cols j≥i
                    gramUpper[idx++] += xi * x[j];                     // Update upper-tri entry
                }
                xty[i] += xi * y;                                      // Update cross-term
            }
            n++;                                                       // Increment count
        }

        // Merge another partial                                                                
        public void merge(GramWritable other) {                        // Merge stats
            if (other.pEff != this.pEff) throw new IllegalArgumentException("Dim mismatch"); // Ensure same dim
            for (int i = 0; i < gramUpper.length; i++) gramUpper[i] += other.gramUpper[i];   // Sum Gram
            for (int i = 0; i < pEff; i++) xty[i] += other.xty[i];                            // Sum cross-term
            this.n += other.n;                                                               // Sum counts
        }

        // Build full symmetric matrix and add ridge alpha on diagonal                         
        public RealMatrix toMatrixWithRidge(double alpha) {            // Convert to A = X^T X + αI
            double[][] A = new double[pEff][pEff];                     // Allocate full matrix
            int idx = 0;                                               // Reset packed cursor
            for (int i = 0; i < pEff; i++) {                           // Fill upper triangle
                for (int j = i; j < pEff; j++) {
                    double v = gramUpper[idx++];                       // Read packed value
                    A[i][j] = v;                                       // Set upper
                    if (i != j) A[j][i] = v;                           // Mirror to lower
                }
            }
            for (int d = 0; d < pEff; d++) A[d][d] += alpha;           // Add ridge α to diagonal
            return new Array2DRowRealMatrix(A, false);                 // Wrap as RealMatrix (no copy)
        }

        public RealVector bVector() { return new ArrayRealVector(xty, false); } // Return X^T y as vector

        @Override public void write(DataOutput out) throws IOException { // Serialize
            out.writeInt(pEff);                                         // Write dim
            out.writeLong(n);                                           // Write count
            out.writeInt(gramUpper.length);                             // Write packed length
            for (double v : gramUpper) out.writeDouble(v);              // Write packed Gram
            out.writeInt(xty.length);                                   // Write vector length
            for (double v : xty) out.writeDouble(v);                    // Write cross-term
        }

        @Override public void readFields(DataInput in) throws IOException { // Deserialize
            this.pEff = in.readInt();                                   // Read dim
            this.n = in.readLong();                                     // Read count
            int gLen = in.readInt();                                    // Read packed length
            this.gramUpper = new double[gLen];                          // Allocate packed
            for (int i = 0; i < gLen; i++) gramUpper[i] = in.readDouble(); // Read values
            int bLen = in.readInt();                                    // Read cross length
            this.xty = new double[bLen];                                // Allocate cross
            for (int i = 0; i < bLen; i++) xty[i] = in.readDouble();    // Read values
        }
    }

    // Mapper for training (skips test fold rows; accumulates Gram stats)                 // Train mapper role
    public static class TrainMapper extends Mapper<LongWritable, Text, Text, GramWritable> { // Mapper signature
        private static final Text KEY = new Text("ridge");              // Single reduce key
        private GramWritable acc;                                       // Per-mapper accumulator
        private int pRaw, pEff;                                         // Raw and effective dims
        private boolean addIntercept;                                   // Intercept flag
        private char sep;                                               // CSV separator
        private int kFolds, testFold;                                   // CV params

        @Override protected void setup(Context ctx) {                   // Setup called once
            Configuration conf = ctx.getConfiguration();                // Get job conf
            this.pRaw = conf.getInt("ridge.pRaw", -1);                  // Read pRaw
            if (pRaw <= 0) throw new RuntimeException("ridge.pRaw must be > 0"); // Validate
            this.addIntercept = conf.getBoolean("ridge.addIntercept", true);     // Read flag
            this.pEff = addIntercept ? (pRaw + 1) : pRaw;               // Compute effective dim
            String s = conf.get("csv.sep", ",");                        // Get separator
            this.sep = (s == null || s.isEmpty()) ? ',' : s.charAt(0);  // Default comma
            this.kFolds = conf.getInt("cv.k", 0);                       // Read K
            this.testFold = conf.getInt("cv.test.fold", -1);            // Read test fold index
            this.acc = new GramWritable(pEff);                          // Init accumulator
        }

        @Override protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException { // Map per line
            String line = value.toString().trim();                      // Get line
            if (line.isEmpty()) return;                                 // Skip empty

            // Skip TEST rows (we train on the remaining K-1 folds)                                  
            if (kFolds > 1 && testFold >= 0) {                          // CV enabled?
                int fold = Math.floorMod(Long.hashCode(key.get()), kFolds); // Hash offset to fold id
                if (fold == testFold) return;                           // Skip test rows in training
            }

            String[] parts = line.split("\\s*\\" + sep + "\\s*");       // Split by sep
            if (parts.length != pRaw + 1) return;                       // Expect y + pRaw
            try {                                                        // Parse guarded
                double y = Double.parseDouble(parts[0]);                // Parse target
                double[] x = new double[pEff];                          // Create feature vector
                int off = 1;                                            // Start after y
                if (addIntercept) {                                     // If intercept
                    x[0] = 1.0;                                         // x0 = 1
                    for (int j = 0; j < pRaw; j++) {                    // For each raw feature
                        double v = Double.parseDouble(parts[off++]);    // Parse feature
                        if (Double.isNaN(v) || Double.isInfinite(v)) return; // Drop invalid
                        x[j + 1] = v;                                   // Shifted index with intercept
                    }
                } else {                                                // No intercept
                    for (int j = 0; j < pRaw; j++) {                    // For each feature
                        double v = Double.parseDouble(parts[off++]);    // Parse feature
                        if (Double.isNaN(v) || Double.isInfinite(v)) return; // Drop invalid
                        x[j] = v;                                       // Store as-is
                    }
                }
                if (Double.isNaN(y) || Double.isInfinite(y)) return;    // Drop invalid y
                acc.add(x, y);                                          // Accumulate stats
            } catch (NumberFormatException ignored) {}                  // Skip malformed line
        }

        @Override protected void cleanup(Context ctx) throws IOException, InterruptedException { // Cleanup once
            ctx.write(KEY, acc);                                        // Emit partial Gram
        }
    }

    // Combiner merges Gram partials to reduce shuffle                                          
    public static class TrainCombiner extends Reducer<Text, GramWritable, Text, GramWritable> { // Combiner
        @Override protected void reduce(Text key, Iterable<GramWritable> vals, Context ctx) throws IOException, InterruptedException { // Reduce
            GramWritable sum = null;                                   // Local sum
            for (GramWritable g : vals) {                              // For each partial
                if (sum == null) { sum = new GramWritable(g.dim()); sum.merge(g); } // Seed + merge
                else { sum.merge(g); }                                 // Merge next
            }
            if (sum != null) ctx.write(key, sum);                      // Emit combined partial
        }
    }

    // Reducer solves (X^T X + αI) w = X^T y and writes weights (needed by eval)                 
    public static class TrainReducer extends Reducer<Text, GramWritable, Text, Text> { // Final reducer
        @Override protected void reduce(Text key, Iterable<GramWritable> vals, Context ctx) throws IOException, InterruptedException { // Reduce to model
            Configuration conf = ctx.getConfiguration();               // Conf
            double alpha = conf.getFloat("ridge.alpha", 1.0f);         // Ridge α

            GramWritable sum = null;                                   // Accumulator
            for (GramWritable g : vals) {                              // Merge all map/combiner outputs
                if (sum == null) { sum = new GramWritable(g.dim()); sum.merge(g); } // Seed
                else { sum.merge(g); }                                 // Merge next
            }
            if (sum == null) throw new IOException("No Gram stats received."); // Safety

            RealMatrix A = sum.toMatrixWithRidge(alpha);               // A = X^T X + αI
            RealVector b = sum.bVector();                              // b = X^T y
            RealVector w;                                              // Solution vector
            try {                                                      // Prefer Cholesky
                w = new CholeskyDecomposition(A, 1e-10, 1e-10).getSolver().solve(b); // Solve
            } catch (Exception e) {                                    // Fallback QR
                w = new QRDecomposition(A).getSolver().solve(b);       // Solve
            }

            // Write weights CSV; you don't need to open it—eval loads it internally.                
            StringBuilder sb = new StringBuilder();                    // Build CSV
            for (int i = 0; i < w.getDimension(); i++) {               // For each coeff
                if (i > 0) sb.append(',');                             // Comma separator
                sb.append(w.getEntry(i));                               // Value
            }
            ctx.write(new Text("weights"), new Text(sb.toString()));   // Output "weights\tw0,..."
        }
    }

    // Eval mapper: keep TEST rows only; emit sums for R² (sumY, sumY2, SSE, n)                  
    public static class EvalMapper extends Mapper<LongWritable, Text, Text, DoubleArrayWritable> { // Eval mapper
        private static final Text KEY = new Text("r2");                // Single key for R^2 accumulation
        private double[] w;                                            // Loaded weights
        private int pRaw, pEff;                                        // Dims
        private boolean addIntercept;                                  // Intercept flag
        private char sep;                                              // CSV separator
        private int kFolds, testFold;                                  // CV params

        @Override protected void setup(Context ctx) throws IOException { // Setup once
            Configuration conf = ctx.getConfiguration();                // Conf
            this.pRaw = conf.getInt("ridge.pRaw", -1);                  // Read pRaw
            if (pRaw <= 0) throw new IOException("ridge.pRaw <= 0");    // Validate
            this.addIntercept = conf.getBoolean("ridge.addIntercept", true); // Intercept?
            this.pEff = addIntercept ? (pRaw + 1) : pRaw;               // Effective dim
            String s = conf.get("csv.sep", ",");                        // CSV sep
            this.sep = (s == null || s.isEmpty()) ? ',' : s.charAt(0);  // Default comma
            this.kFolds = conf.getInt("cv.k", 0);                       // K folds
            this.testFold = conf.getInt("cv.test.fold", -1);            // Test fold

            String wCsv = conf.get("model.weights.csv", null);          // Weights CSV
            if (wCsv == null) throw new IOException("Missing model.weights.csv"); // Required
            String[] parts = wCsv.split(",");                           // Split CSV
            if (parts.length != pEff) throw new IOException("weights length != effective p"); // Validate length
            this.w = new double[pEff];                                  // Allocate
            for (int i = 0; i < pEff; i++) this.w[i] = Double.parseDouble(parts[i]); // Parse weights
        }

        @Override protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException { // Map per line
            String line = value.toString().trim();                      // Get line
            if (line.isEmpty()) return;                                 // Skip empties

            if (kFolds > 1 && testFold >= 0) {                          // If CV
                int fold = Math.floorMod(Long.hashCode(key.get()), kFolds); // Hash to fold
                if (fold != testFold) return;                           // Keep only TEST rows
            }

            String[] parts = line.split("\\s*\\" + sep + "\\s*");       // Split fields
            if (parts.length != pRaw + 1) return;                       // Expect y + pRaw

            try {                                                        // Parse guarded
                double y = Double.parseDouble(parts[0]);                // Target
                double yhat = 0.0;                                      // Prediction
                int off = 1;                                            // Start after y
                if (addIntercept) {                                     // If intercept
                    yhat += w[0];                                       // Add intercept
                    for (int j = 0; j < pRaw; j++) yhat += Double.parseDouble(parts[off++]) * w[j + 1]; // Sum features
                } else {                                                // No intercept
                    for (int j = 0; j < pRaw; j++) yhat += Double.parseDouble(parts[off++]) * w[j];     // Sum features
                }
                double resid = y - yhat;                                // Residual
                ctx.write(KEY, new DoubleArrayWritable(new double[]{y, y * y, resid * resid, 1.0})); // Emit partial sums
            } catch (NumberFormatException ignored) {}                  // Skip bad rows
        }
    }

    // Compact writable for (sumY, sumY2, SSE, n)                                                 
    public static class DoubleArrayWritable implements Writable {      // Tiny array writable
        private double[] a;                                            // Backing array
        public DoubleArrayWritable() {}                                // No-arg ctor
        public DoubleArrayWritable(double[] a) { this.a = a; }         // Value ctor
        public double[] get() { return a; }                            // Accessor
        @Override public void write(DataOutput out) throws IOException { out.writeInt(a.length); for (double v : a) out.writeDouble(v); } // Serialize
        @Override public void readFields(DataInput in) throws IOException { int n=in.readInt(); a=new double[n]; for(int i=0;i<n;i++) a[i]=in.readDouble(); } // Deserialize
    }

    // Eval reducer computes and writes R^2 line                                                   
    public static class EvalReducer extends Reducer<Text, DoubleArrayWritable, Text, Text> { // Reducer for metrics
        @Override protected void reduce(Text key, Iterable<DoubleArrayWritable> vals, Context ctx) throws IOException, InterruptedException { // Aggregate
            double sumY=0, sumY2=0, sse=0, n=0;                        // Accumulators
            for (DoubleArrayWritable daw : vals) {                     // Sum partials
                double[] v = daw.get();                                // Extract 4-tuple
                sumY += v[0]; sumY2 += v[1]; sse += v[2]; n += v[3];   // Accumulate
            }
            double ybar = (n>0) ? sumY/n : 0;                          // Mean y
            double sst = sumY2 - n*ybar*ybar;                          // Total sum of squares
            double r2 = (sst>0) ? (1.0 - sse/sst) : 0.0;               // R^2 with guard
            ctx.write(new Text("R2"),                                  // Output key
                      new Text(String.format(Locale.ROOT, "n=%.0f,sse=%.6f,sst=%.6f,r2=%.6f", n, sse, sst, r2))); // Output value
        }
    }

    // === Helpers to run jobs & read outputs ===                                                   // Driver helpers
    private static int runTrainJob(Configuration base, Path in, Path out, int pRaw, boolean addIntercept, double alpha, char sep, int k, int testFold) throws Exception { // Train job
        Configuration conf = new Configuration(base);                   // Clone conf
        conf.setInt("ridge.pRaw", pRaw);                                // Set pRaw
        conf.setBoolean("ridge.addIntercept", addIntercept);            // Set intercept flag
        conf.setFloat("ridge.alpha", (float)alpha);                     // Set alpha
        conf.set("csv.sep", String.valueOf(sep));                       // Set sep
        conf.setInt("cv.k", k);                                         // Set K
        conf.setInt("cv.test.fold", testFold);                          // Set test fold

        Job job = Job.getInstance(conf, "ridge-train-fold-" + testFold); // New job
        job.setJarByClass(RidgeRegressionCVMR_Intercept.class);         // Jar main class
        job.setMapperClass(TrainMapper.class);                          // Mapper
        job.setMapOutputKeyClass(Text.class);                           // Map key type
        job.setMapOutputValueClass(GramWritable.class);                 // Map value type
        job.setCombinerClass(TrainCombiner.class);                      // Combiner
        job.setReducerClass(TrainReducer.class);                        // Reducer
        job.setOutputKeyClass(Text.class);                              // Final key type
        job.setOutputValueClass(Text.class);                            // Final value type
        FileInputFormat.setInputPaths(job, in);                         // Input path
        FileSystem fs = out.getFileSystem(conf);                        // FS handle
        if (fs.exists(out)) fs.delete(out, true);                       // Cleanup old output
        FileOutputFormat.setOutputPath(job, out);                       // Output path
        job.setNumReduceTasks(1);                                       // Single reducer for one model
        return job.waitForCompletion(true) ? 0 : 1;                     // Run and return code
    }

    private static String readWeightsCSV(Configuration conf, Path modelOut) throws IOException { // Read weights line
        FileSystem fs = modelOut.getFileSystem(conf);                   // FS
        Path part = new Path(modelOut, "part-r-00000");                 // Default reducer file
        try (FSDataInputStream in = fs.open(part);                      // Open HDFS file
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) { // Reader
            String line;                                                // Line buffer
            while ((line = br.readLine()) != null) {                    // Iterate lines
                if (!line.startsWith("weights")) continue;              // Look for weights key
                int tab = line.indexOf('\t');                           // Find tab
                if (tab < 0) continue;                                  // Must have tab
                return line.substring(tab + 1).trim();                  // Return CSV payload
            }
        }
        throw new IOException("weights line not found in " + part);     // If missing
    }

    private static int runEvalJob(Configuration base, Path in, Path out, int pRaw, boolean addIntercept, char sep, int k, int testFold, String weightsCsv) throws Exception { // Eval job
        Configuration conf = new Configuration(base);                   // Clone conf
        conf.setInt("ridge.pRaw", pRaw);                                // Set dims
        conf.setBoolean("ridge.addIntercept", addIntercept);            // Set intercept
        conf.set("csv.sep", String.valueOf(sep));                       // Set sep
        conf.setInt("cv.k", k);                                         // Set K
        conf.setInt("cv.test.fold", testFold);                          // Set fold
        conf.set("model.weights.csv", weightsCsv);                      // Inject weights

        Job job = Job.getInstance(conf, "ridge-eval-fold-" + testFold); // New job
        job.setJarByClass(RidgeRegressionCVMR_Intercept.class);         // Jar class
        job.setMapperClass(EvalMapper.class);                           // Mapper
        job.setMapOutputKeyClass(Text.class);                           // Map key type
        job.setMapOutputValueClass(DoubleArrayWritable.class);          // Map value type
        job.setReducerClass(EvalReducer.class);                         // Reducer
        job.setOutputKeyClass(Text.class);                              // Final key
        job.setOutputValueClass(Text.class);                            // Final value
        FileInputFormat.setInputPaths(job, in);                         // Input path
        FileSystem fs = out.getFileSystem(conf);                        // FS
        if (fs.exists(out)) fs.delete(out, true);                       // Clean old output
        FileOutputFormat.setOutputPath(job, out);                       // Output path
        job.setNumReduceTasks(1);                                       // Single reducer
        return job.waitForCompletion(true) ? 0 : 1;                     // Run and return code
    }

    private static double readFoldR2(Configuration conf, Path evalOut) throws IOException { // Read R^2 from HDFS
        FileSystem fs = evalOut.getFileSystem(conf);                    // FS
        Path part = new Path(evalOut, "part-r-00000");                  // Reducer file
        try (FSDataInputStream in = fs.open(part);                      // Open file
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) { // Reader
            String line;                                                // Line buffer
            while ((line = br.readLine()) != null) {                    // Iterate lines
                if (!line.startsWith("R2")) continue;                   // Find R2 line
                int tab = line.indexOf('\t');                           // Tab index
                if (tab < 0) continue;                                  // Must have tab
                String payload = line.substring(tab + 1);               // Extract "n=...,sse=...,sst=...,r2=..."
                for (String kv : payload.split(",")) {                  // Split fields
                    String[] kvp = kv.split("=");                       // key=value
                    if (kvp.length == 2 && kvp[0].trim().equals("r2")) return Double.parseDouble(kvp[1]); // Return r2
                }
            }
        }
        throw new IOException("R2 not found in " + part);               // If not found
    }

    // Driver: orchestrates K folds, prints per-fold & mean R^2                                             
    @Override
    public int run(String[] args) throws Exception {                    // Tool.run entry
        if (args.length < 9 || args.length > 10) {                     // Validate args
            System.err.println("Usage: <inputCSV> <workDir> <pRaw> <alpha> <kFolds> <csvSep> <jobTag> <addIntercept:true|false> [cleanWorkDir:true|false]"); // Usage
            System.err.println("Example: hdfs:/data/train.csv hdfs:/tmp/ridge 4999 1.0 5 , run1 true true"); // Example
            return 1;                                                  // Exit on error
        }
        Path input = new Path(args[0]);                                // Input CSV (HDFS)
        Path workDir = new Path(args[1]);                              // Work/output dir (HDFS)
        int pRaw = Integer.parseInt(args[2]);                          // Raw feature count
        double alpha = Double.parseDouble(args[3]);                    // Ridge penalty
        int kFolds = Integer.parseInt(args[4]);                        // K folds
        char sep = args[5].charAt(0);                                  // CSV separator
        String tag = args[6];                                          // Job tag for paths
        boolean addIntercept = Boolean.parseBoolean(args[7]);          // Intercept flag
        boolean clean = (args.length == 10) ? Boolean.parseBoolean(args[8]) : true; // Clean workDir?

        Configuration base = getConf();                                // Base conf
        FileSystem fs = workDir.getFileSystem(base);                   // FS for workDir
        if (clean && fs.exists(workDir)) fs.delete(workDir, true);     // Clean previous outputs
        if (!fs.exists(workDir)) fs.mkdirs(workDir);                   // Ensure dir exists

        List<Double> r2s = new ArrayList<>();                          // Collect per-fold R^2
        int K = Math.max(2, kFolds);                                   // Ensure at least 2 folds

        for (int testFold = 0; testFold < K; testFold++) {             // For each fold
            // Train (exclude test fold)                                                                  
            Path modelOut = new Path(workDir, String.format(Locale.ROOT, "model_%s_fold_%d", tag, testFold)); // Model path
            int rc1 = runTrainJob(base, input, modelOut, pRaw, addIntercept, alpha, sep, K, testFold); // Train job
            if (rc1 != 0) return rc1;                                  // Abort on failure

            // Read weights                                                                                
            String wCsv = readWeightsCSV(base, modelOut);              // Load weights CSV (for eval)

            // Eval on test fold                                                                           
            Path evalOut = new Path(workDir, String.format(Locale.ROOT, "eval_%s_fold_%d", tag, testFold)); // Eval path
            int rc2 = runEvalJob(base, input, evalOut, pRaw, addIntercept, sep, K, testFold, wCsv); // Eval job
            if (rc2 != 0) return rc2;                                  // Abort on failure

            // Parse & print R^2                                                                           
            double r2 = readFoldR2(base, evalOut);                     // Parse r2 from eval output
            r2s.add(r2);                                               // Save fold r2
            System.out.printf(Locale.ROOT, "Fold %d R^2 = %.6f%n", testFold, r2); // Print fold r2
        }

        // Mean R^2                                                                                         
        double mean = 0.0;                                             // Init sum
        for (double v : r2s) mean += v;                                // Sum r2s
        mean /= r2s.size();                                            // Average
        System.out.printf(Locale.ROOT, "Mean R^2 over %d folds = %.6f%n", r2s.size(), mean); // Print mean r2
        return 0;                                                      // Success
    }

    public static void main(String[] args) throws Exception {          // Program entry point
        int rc = ToolRunner.run(new Configuration(), new RidgeRegressionCVMR_Intercept(), args); // Run Tool
        System.exit(rc);                                               // Exit with code
    }
}
