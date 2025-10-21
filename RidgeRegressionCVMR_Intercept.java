// File: RidgeRegressionCVMR_Intercept.java // File name for the Java source
// Hadoop MapReduce Ridge Regression with K-fold CV and optional intercept. // High-level description
// Solves (X^T X + alpha I) w = X^T y. // Objective solved by the reducer
// Input rows: dense CSV "y,f1,f2,...,fp" (NO intercept in CSV). // Expected input format
// Set addIntercept=true to include an intercept column internally (x0 = 1). // Intercept behavior

import org.apache.hadoop.conf.Configuration; // Hadoop configuration API
import org.apache.hadoop.conf.Configured;    // Base class for Tool pattern
import org.apache.hadoop.fs.*;               // HDFS filesystem classes
import org.apache.hadoop.io.*;               // Hadoop IO (Writable, etc.)
import org.apache.hadoop.mapreduce.*;        // MapReduce core classes
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;  // Input path helpers
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat; // Output path helpers
import org.apache.hadoop.util.Tool;          // Tool interface for command-line jobs
import org.apache.hadoop.util.ToolRunner;    // Utility to run Tool implementations

import java.io.*;                            // Java IO streams/readers
import java.nio.charset.StandardCharsets;    // Charset for reading text files
import java.util.*;                          // Collections utilities
import java.util.Locale;                     // Locale for formatting strings

import org.apache.commons.math3.linear.*;    // Apache Commons Math linear algebra (Cholesky/QR)

public class RidgeRegressionCVMR_Intercept extends Configured implements Tool { // Main class implementing Tool

    // ===== Writable for upper-triangular Gram and X^T y ===== // Section header for custom Writable
    public static class GramWritable implements Writable { // Custom Writable to carry Gram stats
        private int pEff;            // effective dimension (p + 1 if intercept, else p)
        private double[] gramUpper;  // packed upper triangle of X^T X (size pEff*(pEff+1)/2)
        private double[] xty;        // cross term X^T y (length pEff)
        private long n;              // number of rows accumulated (optional diagnostic)

        public GramWritable() {}     // Empty constructor required by Hadoop

        public GramWritable(int pEff) { // Constructor with effective dimension
            this.pEff = pEff;                                               // Store effective dimension
            this.gramUpper = new double[pEff * (pEff + 1) / 2];             // Allocate packed upper-tri matrix
            this.xty = new double[pEff];                                    // Allocate cross vector
            this.n = 0L;                                                    // Initialize row count
        }

        public int dim() { return pEff; } // Helper to expose dimension

        public void add(double[] x, double y) { // Accumulate a single sample into Gram stats
            int idx = 0;                                                // Cursor for packed upper triangle
            for (int i = 0; i < pEff; i++) {                            // Loop over rows i
                double xi = x[i];                                       // Cache x_i
                for (int j = i; j < pEff; j++) {                        // Loop over cols j >= i
                    gramUpper[idx++] += xi * x[j];                      // Add to (i,j) entry in packed form
                }
                xty[i] += xi * y;                                       // Add to cross term for index i
            }
            n++;                                                        // Increment row counter
        }

        public void merge(GramWritable other) { // Merge another partial GramWritable
            if (other.pEff != this.pEff) throw new IllegalArgumentException("Dim mismatch"); // Sanity check
            for (int i = 0; i < gramUpper.length; i++) gramUpper[i] += other.gramUpper[i];   // Sum packed upper triangles
            for (int i = 0; i < pEff; i++) xty[i] += other.xty[i];                           // Sum cross terms
            this.n += other.n;                                                                // Sum counts
        }

        public RealMatrix toMatrixWithRidge(double alpha) { // Convert to full matrix and add ridge αI
            double[][] A = new double[pEff][pEff];                         // Allocate full symmetric matrix
            int idx = 0;                                                   // Reset packed index
            for (int i = 0; i < pEff; i++) {                               // For each row
                for (int j = i; j < pEff; j++) {                           // For each col j>=i
                    double v = gramUpper[idx++];                           // Read packed value
                    A[i][j] = v;                                           // Set upper entry
                    if (i != j) A[j][i] = v;                               // Mirror to lower entry
                }
            }
            for (int d = 0; d < pEff; d++) A[d][d] += alpha;               // Add α to diagonal for ridge
            return new Array2DRowRealMatrix(A, false);                      // Wrap in RealMatrix (no copy)
        }

        public RealVector bVector() { return new ArrayRealVector(xty, false); } // Return cross term as vector

        @Override public void write(DataOutput out) throws IOException { // Serialize to DataOutput
            out.writeInt(pEff);                                          // Write dimension
            out.writeLong(n);                                            // Write row count
            out.writeInt(gramUpper.length);                              // Write packed length
            for (double v : gramUpper) out.writeDouble(v);               // Write packed entries
            out.writeInt(xty.length);                                    // Write cross-term length
            for (double v : xty) out.writeDouble(v);                     // Write cross-term values
        }

        @Override public void readFields(DataInput in) throws IOException { // Deserialize from DataInput
            this.pEff = in.readInt();                                     // Read dimension
            this.n = in.readLong();                                       // Read row count
            int gLen = in.readInt();                                      // Read packed length
            this.gramUpper = new double[gLen];                             // Allocate packed array
            for (int i = 0; i < gLen; i++) gramUpper[i] = in.readDouble(); // Read packed values
            int bLen = in.readInt();                                      // Read cross-term length
            this.xty = new double[bLen];                                   // Allocate cross-term array
            for (int i = 0; i < bLen; i++) xty[i] = in.readDouble();       // Read cross-term values
        }
    }

    // ===== Train mapper: build (X^T X, X^T y) from TRAIN rows only ===== // Section header for the training mapper
    public static class TrainMapper extends Mapper<LongWritable, Text, Text, GramWritable> { // Mapper for training fold
        private static final Text KEY = new Text("ridge"); // Single reduce key to funnel all stats
        private GramWritable acc;                          // Per-mapper aggregator for Gram stats
        private int pRaw;             // number of raw features (no intercept) e.g., 4999
        private int pEff;             // effective dimension: pRaw (+1 if intercept)
        private boolean addIntercept; // whether to prepend x0=1
        private char sep;             // CSV separator
        private int kFolds;           // number of folds (>=2 for CV)
        private int testFold;         // index of held-out fold for this run

        @Override protected void setup(Context ctx) { // Setup called once per mapper task
            Configuration conf = ctx.getConfiguration();                        // Access job configuration
            this.pRaw = conf.getInt("ridge.pRaw", -1);                          // Read pRaw from conf
            if (pRaw <= 0) throw new RuntimeException("ridge.pRaw must be >0"); // Validate pRaw
            this.addIntercept = conf.getBoolean("ridge.addIntercept", true);    // Read addIntercept flag
            this.pEff = addIntercept ? (pRaw + 1) : pRaw;                        // Compute effective dimension
            String s = conf.get("csv.sep", ",");                                 // Read CSV separator
            this.sep = (s == null || s.isEmpty()) ? ',' : s.charAt(0);           // Default to comma
            this.kFolds = conf.getInt("cv.k", 0);                                // Number of CV folds
            this.testFold = conf.getInt("cv.test.fold", -1);                     // Test fold index
            this.acc = new GramWritable(pEff);                                   // Allocate aggregator
        }

        @Override protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException { // Map per input line
            String line = value.toString().trim();                          // Read and trim the input line
            if (line.isEmpty()) return;                                     // Skip empty lines

            if (kFolds > 1 && testFold >= 0) {                              // If doing CV with a valid test fold
                int fold = Math.floorMod(Long.hashCode(key.get()), kFolds); // Stable hash of byte offset to fold
                if (fold == testFold) return;                               // Skip test rows during training
            }

            String[] parts = line.split("\\s*\\" + sep + "\\s*");           // Split CSV by configured separator
            if (parts.length != pRaw + 1) return;                            // Expect exactly y + pRaw values
            try {                                                            // Parse guarded by try/catch
                double y = Double.parseDouble(parts[0]);                     // Parse response y
                double[] x = new double[pEff];                               // Allocate feature vector
                int off = 1;                                                 // Offset into parts for features
                if (addIntercept) {                                          // If intercept is enabled
                    x[0] = 1.0;                                              // Set x0 = 1
                    for (int j = 0; j < pRaw; j++) {                         // Loop through raw features
                        double v = Double.parseDouble(parts[off++]);         // Parse feature value
                        if (Double.isNaN(v) || Double.isInfinite(v)) return; // Drop row on invalid numeric
                        x[j + 1] = v;                                        // Store at shifted index
                    }
                } else {                                                     // If intercept not enabled
                    for (int j = 0; j < pRaw; j++) {                         // Loop through raw features
                        double v = Double.parseDouble(parts[off++]);         // Parse feature value
                        if (Double.isNaN(v) || Double.isInfinite(v)) return; // Drop row on invalid numeric
                        x[j] = v;                                            // Store at same index
                    }
                }
                if (Double.isNaN(y) || Double.isInfinite(y)) return;         // Drop row if y invalid
                acc.add(x, y);                                               // Accumulate into Gram stats
            } catch (NumberFormatException ignored) {}                        // Silently skip malformed rows
        }

        @Override protected void cleanup(Context ctx) throws IOException, InterruptedException { // Cleanup called once per mapper
            ctx.write(KEY, acc);                                             // Emit the partial Gram stats
        }
    }

    // ===== Combiner: merge Gram partials ===== // Section header for combiner
    public static class TrainCombiner extends Reducer<Text, GramWritable, Text, GramWritable> { // Combiner reduces map output
        @Override protected void reduce(Text key, Iterable<GramWritable> vals, Context ctx) throws IOException, InterruptedException { // Reduce partials
            GramWritable sum = null;                                   // Start with no accumulator
            for (GramWritable g : vals) {                              // For each mapper-emitted partial
                if (sum == null) { sum = new GramWritable(g.dim()); sum.merge(g); } // Initialize and merge first
                else { sum.merge(g); }                                 // Merge subsequent partials
            }
            if (sum != null) ctx.write(key, sum);                      // Emit combined partial to reducer
        }
    }

    // ===== Reducer: solve (X^T X + αI) w = X^T y ===== // Section header for reducer
    public static class TrainReducer extends Reducer<Text, GramWritable, Text, Text> { // Reducer to finalize model
        @Override protected void reduce(Text key, Iterable<GramWritable> vals, Context ctx) throws IOException, InterruptedException { // Reduce method
            Configuration conf = ctx.getConfiguration();                 // Access configuration
            double alpha = conf.getFloat("ridge.alpha", 1.0f);           // Read ridge α from conf

            GramWritable sum = null;                                     // Accumulator for all mapper/combiner outputs
            for (GramWritable g : vals) {                                // Iterate incoming partials
                if (sum == null) { sum = new GramWritable(g.dim()); sum.merge(g); } // Initialize accumulator
                else { sum.merge(g); }                                   // Merge next partial
            }
            if (sum == null) throw new IOException("No Gram stats received."); // Safety check

            RealMatrix A = sum.toMatrixWithRidge(alpha);                 // Build A = X^T X + αI
            RealVector b = sum.bVector();                                // Build b = X^T y
            RealVector w;                                                // Placeholder for solution vector

            try {                                                        // Attempt Cholesky (SPD expected after ridge)
                w = new CholeskyDecomposition(A, 1e-10, 1e-10).getSolver().solve(b); // Solve via Cholesky
            } catch (Exception e) {                                      // If Cholesky fails
                w = new QRDecomposition(A).getSolver().solve(b);         // Fallback to QR solve
            }

            StringBuilder sb = new StringBuilder();                       // Prepare CSV for weights
            for (int i = 0; i < w.getDimension(); i++) {                  // Loop over coefficients
                if (i > 0) sb.append(',');                                // Add comma separator
                sb.append(w.getEntry(i));                                  // Append coefficient value
            }
            ctx.write(new Text("weights"), new Text(sb.toString()));      // Emit "weights\tw0,w1,...,wp-1"
        }
    }

    // ===== Eval mapper: compute partials for R² on TEST rows only ===== // Section header for evaluation mapper
    public static class EvalMapper extends Mapper<LongWritable, Text, Text, DoubleArrayWritable> { // Mapper for evaluation
        private static final Text KEY = new Text("r2"); // Single key for R² aggregation
        private double[] w;            // Learned weights loaded from conf
        private int pRaw;              // Raw features count (no intercept)
        private int pEff;              // Effective dimension (pRaw + intercept?)
        private boolean addIntercept;  // Whether weights include intercept at w[0]
        private char sep;              // CSV separator
        private int kFolds;            // Number of folds
        private int testFold;          // Test fold index

        @Override protected void setup(Context ctx) throws IOException { // Setup for evaluation mapper
            Configuration conf = ctx.getConfiguration();                          // Get config
            this.pRaw = conf.getInt("ridge.pRaw", -1);                            // Read pRaw
            if (pRaw <= 0) throw new IOException("ridge.pRaw <= 0");              // Validate
            this.addIntercept = conf.getBoolean("ridge.addIntercept", true);      // Read intercept flag
            this.pEff = addIntercept ? (pRaw + 1) : pRaw;                          // Compute effective dimension
            String s = conf.get("csv.sep", ",");                                   // Read CSV sep
            this.sep = (s == null || s.isEmpty()) ? ',' : s.charAt(0);             // Default comma
            this.kFolds = conf.getInt("cv.k", 0);                                  // Read K
            this.testFold = conf.getInt("cv.test.fold", -1);                       // Read test fold

            String wCsv = conf.get("model.weights.csv", null);                     // Read weights as CSV string
            if (wCsv == null) throw new IOException("Missing model.weights.csv");  // Ensure present
            String[] parts = wCsv.split(",");                                      // Split weights by commas
            if (parts.length != pEff) throw new IOException("weights length != effective p"); // Validate length
            this.w = new double[pEff];                                             // Allocate weight array
            for (int i = 0; i < pEff; i++) this.w[i] = Double.parseDouble(parts[i]); // Parse each weight
        }

        @Override protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException { // Evaluate per line
            String line = value.toString().trim();                               // Read and trim line
            if (line.isEmpty()) return;                                          // Skip empties

            if (kFolds > 1 && testFold >= 0) {                                   // If cross-validating
                int fold = Math.floorMod(Long.hashCode(key.get()), kFolds);      // Hash offset to fold
                if (fold != testFold) return;                                    // Keep only TEST fold rows
            }

            String[] parts = line.split("\\s*\\" + sep + "\\s*");                // Split CSV
            if (parts.length != pRaw + 1) return;                                 // Expect y + pRaw

            try {                                                                 // Parse computations
                double y = Double.parseDouble(parts[0]);                          // Parse response
                double yhat = 0.0;                                               // Init prediction
                int off = 1;                                                     // Feature offset
                if (addIntercept) {                                              // If intercept enabled
                    yhat += 1.0 * w[0];                                          // Add intercept term
                    for (int j = 0; j < pRaw; j++) {                             // Loop raw features
                        double xj = Double.parseDouble(parts[off++]);            // Parse feature
                        yhat += xj * w[j + 1];                                   // Accumulate x·w
                    }
                } else {                                                          // If no intercept
                    for (int j = 0; j < pRaw; j++) {                              // Loop features
                        double xj = Double.parseDouble(parts[off++]);             // Parse feature
                        yhat += xj * w[j];                                        // Accumulate x·w
                    }
                }
                double resid = y - yhat;                                          // Residual y - ŷ
                double resid2 = resid * resid;                                    // Square residual
                double y2 = y * y;                                                // y squared
                ctx.write(KEY, new DoubleArrayWritable(new double[]{y, y2, resid2, 1.0})); // Emit partial sums
            } catch (NumberFormatException ignored) {}                             // Skip malformed rows
        }
    }

    // ===== Writable for (sumY, sumY2, SSE, count) ===== // Section header for compact vector writable
    public static class DoubleArrayWritable implements Writable { // Simple custom Writable for 4 doubles
        private double[] a;                              // Backing array
        public DoubleArrayWritable() {}                 // Empty constructor
        public DoubleArrayWritable(double[] a) { this.a = a; } // Construct with array
        public double[] get() { return a; }             // Accessor for reducer
        @Override public void write(DataOutput out) throws IOException { out.writeInt(a.length); for (double v : a) out.writeDouble(v); } // Serialize array
        @Override public void readFields(DataInput in) throws IOException { int n=in.readInt(); a=new double[n]; for(int i=0;i<n;i++) a[i]=in.readDouble(); } // Deserialize array
    }

    // ===== Eval reducer: compute R² ===== // Section header for evaluation reducer
    public static class EvalReducer extends Reducer<Text, DoubleArrayWritable, Text, Text> { // Reducer computing R²
        @Override protected void reduce(Text key, Iterable<DoubleArrayWritable> vals, Context ctx) throws IOException, InterruptedException { // Reduce over partials
            double sumY=0, sumY2=0, sse=0, n=0;                 // Initialize accumulators
            for (DoubleArrayWritable daw : vals) {              // Iterate all mapper outputs
                double[] v = daw.get();                         // Get four-tuple
                sumY += v[0]; sumY2 += v[1]; sse += v[2]; n += v[3]; // Accumulate sums
            }
            double ybar = (n>0) ? sumY/n : 0;                   // Compute mean of y
            double sst = sumY2 - n*ybar*ybar;                   // Total sum of squares
            double r2 = (sst>0) ? (1.0 - sse/sst) : 0.0;        // R² = 1 - SSE/SST (guard SST=0)
            ctx.write(new Text("R2"),                           // Emit key "R2"
                     new Text(String.format(Locale.ROOT, "n=%.0f,sse=%.6f,sst=%.6f,r2=%.6f", n, sse, sst, r2))); // Emit metrics
        }
    }

    // ===== Helpers to run jobs and read outputs ===== // Section header for driver helper methods
    private static int runTrainJob(Configuration base, Path in, Path out, int pRaw, boolean addIntercept, double alpha, char sep, int k, int testFold) throws Exception { // Launch train job
        Configuration conf = new Configuration(base);                             // Copy base configuration
        conf.setInt("ridge.pRaw", pRaw);                                          // Set raw feature count
        conf.setBoolean("ridge.addIntercept", addIntercept);                      // Set intercept flag
        conf.setFloat("ridge.alpha", (float)alpha);                               // Set ridge alpha
        conf.set("csv.sep", String.valueOf(sep));                                  // Set CSV separator
        conf.setInt("cv.k", k);                                                   // Set number of folds
        conf.setInt("cv.test.fold", testFold);                                    // Set current test fold

        Job job = Job.getInstance(conf, "ridge-train-fold-" + testFold);          // Create new MR job
        job.setJarByClass(RidgeRegressionCVMR_Intercept.class);                   // Set job JAR main class
        job.setMapperClass(TrainMapper.class);                                     // Mapper class
        job.setMapOutputKeyClass(Text.class);                                      // Mapper output key type
        job.setMapOutputValueClass(GramWritable.class);                            // Mapper output value type
        job.setCombinerClass(TrainCombiner.class);                                 // Combiner class to reduce shuffle
        job.setReducerClass(TrainReducer.class);                                   // Reducer class
        job.setOutputKeyClass(Text.class);                                         // Final output key type
        job.setOutputValueClass(Text.class);                                       // Final output value type
        FileInputFormat.setInputPaths(job, in);                                    // Configure input path
        FileSystem fs = out.getFileSystem(conf);                                   // Get FS for output
        if (fs.exists(out)) fs.delete(out, true);                                   // Delete output if exists
        FileOutputFormat.setOutputPath(job, out);                                  // Configure output path
        job.setNumReduceTasks(1);                                                  // Single reducer for one model
        return job.waitForCompletion(true) ? 0 : 1;                                // Run job and return status
    }

    private static String readWeightsCSV(Configuration conf, Path modelOut) throws IOException { // Load weights from reducer output
        FileSystem fs = modelOut.getFileSystem(conf);                   // Get FS
        Path part = new Path(modelOut, "part-r-00000");                 // Default reducer output file
        try (FSDataInputStream in = fs.open(part);                      // Open HDFS file
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) { // Wrap reader
            String line;                                                // Line buffer
            while ((line = br.readLine()) != null) {                    // Read each line
                if (!line.startsWith("weights")) continue;              // Look for weights line
                int tab = line.indexOf('\t');                           // Find tab separator
                if (tab < 0) continue;                                  // Guard malformed line
                return line.substring(tab + 1).trim();                  // Return CSV "w0,w1,...,wp-1"
            }
        }
        throw new IOException("weights line not found in " + part);     // Error if not found
    }

    private static int runEvalJob(Configuration base, Path in, Path out, int pRaw, boolean addIntercept, char sep, int k, int testFold, String weightsCsv) throws Exception { // Launch eval job
        Configuration conf = new Configuration(base);                         // Copy base conf
        conf.setInt("ridge.pRaw", pRaw);                                      // Raw feature count
        conf.setBoolean("ridge.addIntercept", addIntercept);                  // Intercept flag
        conf.set("csv.sep", String.valueOf(sep));                              // CSV separator
        conf.setInt("cv.k", k);                                               // Number of folds
        conf.setInt("cv.test.fold", testFold);                                // Which fold is test
        conf.set("model.weights.csv", weightsCsv);                             // Inject weights for mapper

        Job job = Job.getInstance(conf, "ridge-eval-fold-" + testFold);       // Create eval job
        job.setJarByClass(RidgeRegressionCVMR_Intercept.class);               // Set JAR class
        job.setMapperClass(EvalMapper.class);                                  // Mapper for eval
        job.setMapOutputKeyClass(Text.class);                                  // Mapper output key
        job.setMapOutputValueClass(DoubleArrayWritable.class);                 // Mapper output value
        job.setReducerClass(EvalReducer.class);                                // Reducer computing R²
        job.setOutputKeyClass(Text.class);                                     // Final output key
        job.setOutputValueClass(Text.class);                                   // Final output value
        FileInputFormat.setInputPaths(job, in);                                // Set input path
        FileSystem fs = out.getFileSystem(conf);                               // FS for output
        if (fs.exists(out)) fs.delete(out, true);                               // Delete old output if exists
        FileOutputFormat.setOutputPath(job, out);                              // Set output path
        job.setNumReduceTasks(1);                                              // Single reducer to aggregate R²
        return job.waitForCompletion(true) ? 0 : 1;                            // Run job and return status
    }

    private static double readFoldR2(Configuration conf, Path evalOut) throws IOException { // Read R² from eval output
        FileSystem fs = evalOut.getFileSystem(conf);               // Get FS
        Path part = new Path(evalOut, "part-r-00000");             // Reducer output file
        try (FSDataInputStream in = fs.open(part);                 // Open file
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) { // Reader
            String line;                                           // Line buffer
            while ((line = br.readLine()) != null) {               // Iterate lines
                if (!line.startsWith("R2")) continue;              // Look for "R2" record
                int tab = line.indexOf('\t');                      // Find tab
                if (tab < 0) continue;                             // Guard
                String payload = line.substring(tab + 1);          // Extract "n=...,sse=...,sst=...,r2=..."
                for (String kv : payload.split(",")) {             // Split by commas
                    String[] kvp = kv.split("=");                  // Split key=value
                    if (kvp.length == 2 && kvp[0].trim().equals("r2")) // Check r2 key
                        return Double.parseDouble(kvp[1]);         // Parse and return r2
                }
            }
        }
        throw new IOException("R2 not found in " + part);          // Error if missing
    }

    // ===== Tool entry: orchestrate K-fold CV ===== // Section header for Tool.run
    @Override
    public int run(String[] args) throws Exception { // Entry point for Tool (parses CLI and orchestrates jobs)
        if (args.length < 9 || args.length > 10) { // Validate argument count
            System.err.println("Usage: <inputCSV> <workDir> <pRaw> <alpha> <kFolds> <csvSep> <jobTag> <addIntercept:true|false> [cleanWorkDir:true|false]"); // Usage string
            System.err.println("Example: hdfs:/data/train.csv hdfs:/tmp/ridge 4999 1.0 5 , run1 true true"); // Example invocation
            return 1; // Non-zero exit on usage error
        }

        Path input = new Path(args[0]);             // HDFS path to input CSV
        Path workDir = new Path(args[1]);           // HDFS working/output directory
        int pRaw = Integer.parseInt(args[2]);       // Raw feature count (no intercept)
        double alpha = Double.parseDouble(args[3]); // Ridge penalty α
        int kFolds = Integer.parseInt(args[4]);     // Number of CV folds
        char sep = args[5].charAt(0);               // CSV separator character
        String tag = args[6];                       // Tag to differentiate outputs
        boolean addIntercept = Boolean.parseBoolean(args[7]); // Whether to add intercept internally
        boolean clean = (args.length == 10) ? Boolean.parseBoolean(args[8]) : true; // Clean workDir flag

        Configuration base = getConf();                   // Base Hadoop configuration
        FileSystem fs = workDir.getFileSystem(base);      // FileSystem for workDir
        if (clean && fs.exists(workDir)) fs.delete(workDir, true); // Optionally remove existing workDir
        if (!fs.exists(workDir)) fs.mkdirs(workDir);      // Ensure workDir exists

        List<Double> r2s = new ArrayList<>();             // Collect per-fold R²
        int K = Math.max(2, kFolds);                      // Ensure at least 2 folds

        for (int testFold = 0; testFold < K; testFold++) { // Loop over folds
            Path modelOut = new Path(workDir, String.format(Locale.ROOT, "model_%s_fold_%d", tag, testFold)); // Model output path
            int rc1 = runTrainJob(base, input, modelOut, pRaw, addIntercept, alpha, sep, K, testFold); // Train fold
            if (rc1 != 0) return rc1; // Abort on failure

            String wCsv = readWeightsCSV(base, modelOut); // Read trained weights CSV

            Path evalOut = new Path(workDir, String.format(Locale.ROOT, "eval_%s_fold_%d", tag, testFold)); // Eval output path
            int rc2 = runEvalJob(base, input, evalOut, pRaw, addIntercept, sep, K, testFold, wCsv); // Evaluate fold
            if (rc2 != 0) return rc2; // Abort on failure

            double r2 = readFoldR2(base, evalOut); // Parse R² from output
            r2s.add(r2);                           // Save fold R²
            System.out.printf(Locale.ROOT, "Fold %d R^2 = %.6f%n", testFold, r2); // Print per-fold R²
        }

        double mean = 0.0;                    // Initialize mean accumulator
        for (double v : r2s) mean += v;       // Sum per-fold R²
        mean /= r2s.size();                   // Compute mean R²
        System.out.printf(Locale.ROOT, "Mean R^2 over %d folds = %.6f%n", r2s.size(), mean); // Print mean R²
        return 0; // Success
    }

    public static void main(String[] args) throws Exception { // Standard Java entry point delegates to ToolRunner
        int rc = ToolRunner.run(new Configuration(), new RidgeRegressionCVMR_Intercept(), args); // Launch Tool
        System.exit(rc); // Exit with Tool return code
    }
}
