import java.io.*;
import com.qcri.farasa.diacritize.DiacritizeText;
import com.qcri.farasa.segmenter.Farasa;
import com.qcri.farasa.pos.FarasaPOSTagger;

public class DatasetDiacritizer {
    public static void main(String[] args) throws Exception {
        // Set paths
        String inputFilePath = "../../data/raw/apcd/apcd_full.csv";
        String outputFilePath = "../../data/raw/apcd/apcd_full_diacritized.csv";

        // Initialize Farasa components
        Farasa farasa = new Farasa();
        FarasaPOSTagger farasaPOSTagger = new FarasaPOSTagger(farasa);
        DiacritizeText diacritizer = new DiacritizeText(farasa, farasaPOSTagger);

        // Read the input CSV file
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(inputFilePath), "UTF-8"));
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFilePath), "UTF-8"));

        // Read and process the CSV file
        String line = reader.readLine(); // Read header line
        if (line != null) {
            writer.write(line); // Write header to output file
            writer.newLine();
        }

        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) {
                String[] columns = line.split(","); // Split CSV line into columns
                String lastColumn = columns[columns.length - 1]; // Get the last column (البيت)

                // Diacritize the last column
                String diacritizedLastColumn = diacritizer.diacritize(lastColumn.trim());

                // Replace the last column with the diacritized version
                columns[columns.length - 1] = diacritizedLastColumn;

                // Reassemble the line and write to output file
                writer.write(String.join(",", columns));
                writer.newLine();
            }
        }

        // Close resources
        reader.close();
        writer.close();

        // Cleanup Farasa resources
        System.out.println("Diacritization complete! Output saved to: " + outputFilePath);
    }
}
