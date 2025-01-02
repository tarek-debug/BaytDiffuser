import java.io.*;
import com.qcri.farasa.diacritize.DiacritizeText;
import com.qcri.farasa.segmenter.Farasa;
import com.qcri.farasa.pos.FarasaPOSTagger;

public class VerseDiacritizer {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Please provide an Arabic text input.");
            return;
        }

        // Combine arguments to form the input text
        String inputText = String.join(" ", args);

        // Initialize Farasa components
        Farasa farasa = new Farasa();
        FarasaPOSTagger farasaPOSTagger = new FarasaPOSTagger(farasa);
        DiacritizeText diacritizer = new DiacritizeText(farasa, farasaPOSTagger);

        // Perform diacritization
        String diacritizedText = diacritizer.diacritize(inputText.trim());

        // Output the diacritized text as UTF-8
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(System.out, "UTF-8"), true);
        writer.println(diacritizedText);
    }
}
