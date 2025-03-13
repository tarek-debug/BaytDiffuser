import javax.swing.*;
import com.qcri.farasa.diacritize.DiacritizeText;
import com.qcri.farasa.segmenter.Farasa;
import com.qcri.farasa.pos.FarasaPOSTagger;

public class VerseDiacritizerGUI {
    public static void main(String[] args) throws Exception {
        // Create input dialog
        String inputText = JOptionPane.showInputDialog("Enter Arabic text:");

        if (inputText == null || inputText.trim().isEmpty()) {
            JOptionPane.showMessageDialog(null, "No input provided!");
            return;
        }

        // Initialize Farasa components
        Farasa farasa = new Farasa();
        FarasaPOSTagger farasaPOSTagger = new FarasaPOSTagger(farasa);
        DiacritizeText diacritizer = new DiacritizeText(farasa, farasaPOSTagger);

        // Perform diacritization
        String diacritizedText = diacritizer.diacritize(inputText.trim());

        // Display output in a message dialog
        JOptionPane.showMessageDialog(null, "Diacritized Text:\n" + diacritizedText);
    }
}
