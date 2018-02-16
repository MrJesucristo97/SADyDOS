import java.io.File;
import java.io.IOException;



import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class pVoluntaria {
	
	/**
	 * este método convienrte un directorio contenedor de 
	 * texto plano en un set de instancias .arff
	 * estas instancias se guardarán en el path especificado
	 * y se devolverán cmo parametro
	 * @param pathLoad
	 * @param pathSave
	 */
	public Instances converterDirectoryPlainTextToArff(String pathLoad,String pathSave) {
		
		File fi=new File(pathLoad);	
		TextDirectoryLoader loader=new TextDirectoryLoader();
		Instances data=null;
		try {
			//cargar directorio 
			loader.setSource(fi);
			 data=loader.getDataSet();
			
			//guardar archivo (.arff)
			fi=new File(pathSave);
			ArffSaver saver =new ArffSaver();
			saver.setInstances(data);
			saver.setFile(fi);
			saver.writeBatch();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		return data;
		}

	/**
	 * filtrado de texto para lograr discriminar 
	 * posibles atributos para la predicción
	 * @param aFiltrar
	 * @return
	 */
	public Instances filtradoStringWordVector(Instances aFiltrar) {
		Instances firstFilter=null;
		// Creamos el filtro (StringWordVector)
		try {
			StringToWordVector filtroA = new StringToWordVector();
			filtroA.setInputFormat(aFiltrar);
			 firstFilter = Filter.useFilter(aFiltrar, filtroA);
		}catch(Exception e) {

		}
		return firstFilter;


	}
	
	/**
	 * filtrado de atributos quedandonos aquelos con
	 * mayor correlación
	 * @param aFiltrar
	 * @return
	 */
	public Instances filtradoAtributeSelection(Instances aFiltrar) {
		Instances secondFilter=null;
		try {
			// Creamos el filtro (AttributteSelection)
			AttributeSelection filtroB = new AttributeSelection();
			CfsSubsetEval eval = new CfsSubsetEval();
			BestFirst search = new BestFirst();
			filtroB.setEvaluator(eval);
			filtroB.setSearch(search);
			filtroB.setInputFormat(aFiltrar);
			// Filtramos la información
			secondFilter = Filter.useFilter(aFiltrar, filtroB);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return secondFilter;
	}
}
