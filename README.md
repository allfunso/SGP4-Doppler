# SGP4-Doppler
SGP4 orbit propagation plus Doppler tracking for position recalibration

Authors
	
- Alfonso Castro Camino 			
- David Alejandro Araiza Pérez
- Miguel Comett Figueroa 			
- Edgar Ghenno Manrique
- Adriel Santamaria Hernández  			 		
- Fernando Nava Guillén

Description:
The process begins with the reception of the Two-Line Element (TLE) orbital elements, which are propagated using the SGP4 model to predict the satellite's trajectory. From this information, the specific points at which communication will be established or images captured are determined. The obtained coordinates are then converted to CSV format for storage and analysis. In addition, a position recalibration stage is performed using the Doppler effect, which improves the system's accuracy before saving the data to the final file.
