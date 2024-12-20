package org.naomod.yakindu2satecharts;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Collectors;

import org.eclipse.emf.common.EMFPlugin;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.URIConverter;
import org.eclipse.emf.ecore.resource.impl.ExtensibleURIConverterImpl;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.m2m.atl.emftvm.EmftvmFactory;
import org.eclipse.m2m.atl.emftvm.ExecEnv;
import org.eclipse.m2m.atl.emftvm.Metamodel;
import org.eclipse.m2m.atl.emftvm.Model;
import org.eclipse.m2m.atl.emftvm.impl.resource.EMFTVMResourceFactoryImpl;
import org.eclipse.m2m.atl.emftvm.util.DefaultModuleResolver;
import org.eclipse.m2m.atl.emftvm.util.ModuleResolver;
import org.eclipse.m2m.atl.emftvm.util.TimingData;

public class RunTestTransformation {

	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	public static void main(String[] args) throws IOException {

		Map<String, Object> map = Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap();
		map.put("xmi", new XMIResourceFactoryImpl());
		map.put("ecore", new EcoreResourceFactoryImpl());
		map.put("emftvm", new EMFTVMResourceFactoryImpl());

		ExecEnv env = EmftvmFactory.eINSTANCE.createExecEnv();
		ResourceSet rs = new ResourceSetImpl();

		// Register the metamodels into resource set
		EPackage FamiliesPkg = (EPackage) rs.getResource(resourceURI("/../Test/Families.ecore"), true).getContents()
				.get(0);
		EPackage.Registry.INSTANCE.put(FamiliesPkg.getNsURI(), FamiliesPkg);
		EPackage PersonsPkg = (EPackage) rs.getResource(resourceURI("/../Test/Persons.ecore"), true).getContents()
				.get(0);
		EPackage.Registry.INSTANCE.put(PersonsPkg.getNsURI(), PersonsPkg);

		// Load metamodels into execenv
		Metamodel familiesMetaModel = EmftvmFactory.eINSTANCE.createMetamodel();
		familiesMetaModel.setResource(rs.getResource(resourceURI("/../Test/Families.ecore"), true));
		env.registerMetaModel("Families", familiesMetaModel);

		Metamodel personsMetaModel = EmftvmFactory.eINSTANCE.createMetamodel();
		personsMetaModel.setResource(rs.getResource(resourceURI("/../Test/Persons.ecore"), true));
		env.registerMetaModel("Persons", personsMetaModel);

		String relativeInputPath = "../Test/sample-Families.xmi";
		Path absoluteInputPath = Paths.get(relativeInputPath).normalize().toAbsolutePath();
		String relativeTracePath = "../Test/traces.xmi";
		Path absoluteTracePath = Paths.get(relativeTracePath).normalize().toAbsolutePath();
		String relativeOutputPath = "../Test/persons.xmi";
		Path absoluteOutputPath = Paths.get(relativeOutputPath).normalize().toAbsolutePath();
		
		URIConverter fileConverter = createURIConverter();
		rs.setURIConverter(fileConverter);

		// Load models
		URI inputUri = URI.createFileURI(absoluteInputPath.toString());
		Model inModel = EmftvmFactory.eINSTANCE.createModel();
		inModel.setResource(rs.getResource(inputUri, true));
		env.registerInputModel("IN", inModel);

		URI uriTrace = URI.createFileURI(absoluteTracePath.toString());
		Model traceOutModel = EmftvmFactory.eINSTANCE.createModel();
		traceOutModel.setResource(rs.createResource(uriTrace));
		env.registerOutputModel("trace", traceOutModel);

		URI uriOut = URI.createFileURI(absoluteOutputPath.toString());
		Model outModel = EmftvmFactory.eINSTANCE.createModel();
		outModel.setResource(rs.createResource(uriOut));
		env.registerOutputModel("OUT", outModel);

		// Load and run module
		ModuleResolver mr = new DefaultModuleResolver("./../Test/emftvm/", new ResourceSetImpl());
		TimingData td = new TimingData();
		env.loadModule(mr, "Families2Persons");
		td.finishLoading();
		env.run(td);
		td.finish();

		// Save models
		inModel.getResource().save(Collections.emptyMap());
		traceOutModel.getResource().save(Collections.emptyMap());
		outModel.getResource().save(Collections.emptyMap());
	}
	
	protected static URIConverter createURIConverter() {
	    // Converts file:/... URIs to relative URIs
	    // Easier to deal with them in PyEcore
	    return new ExtensibleURIConverterImpl() {

	        @Override
	        public URI normalize(URI uri) {
	            if (uri.isFile()) {
	                URI normalized = getURIMap().get(uri);
	                if (normalized == null) {
	                    //String path = uri.segmentsList().stream().skip(2).collect(Collectors.joining("/"));
	                    
	                    File file = new File(uri.toFileString());

	                    // Check if the file exists
	                    if (file.exists()) {
	                    	// Get the canonical path of the file
	                        String canonicalPath;
	                        try {
	                            canonicalPath = file.getCanonicalPath();
	                        } catch (IOException e) {
	                            canonicalPath = file.getAbsolutePath();
	                        }

	                        normalized = URI.createURI(canonicalPath);
	                        getURIMap().put(uri, normalized);
	                    }
	        
	                }
	                if (normalized != null) {
	                    return normalized;
	                }
	            }
	            return super.normalize(uri);
	        }
	    };
	}
}
