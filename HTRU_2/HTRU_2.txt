  HTRU2 is a data set which describes a sample of pulsar candidates collected during the
	High Time Resolution Universe Survey (South) 
	
	Pulsars are a rare type of Neutron star that produce radio emission detectable here on
	Earth. They are of considerable scientific interest as probes of space-time, the inter-
	stellar medium, and states of matter.
	
	As pulsars rotate, their emission beam sweeps across the sky, and when this crosses
	our line of sight, produces a detectable pattern of broadband radio emission. As pulsars
	rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking
	for periodic radio signals with large radio telescopes.
	
	Each pulsar produces a slightly different emission pattern, which varies slightly with each
	rotation. Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the
	pulsar, as determined by the length of an observation. In the absence of additional info,
	each candidate could potentially describe a real pulsar. However in practice almost all
	detections are caused by radio frequency interference (RFI) and noise, making legitimate
	signals hard to find.
	
	Machine learning tools are now being used to automatically label pulsar candidates to
	facilitate rapid analysis. Classification systems in particular are being widely adopted,
	(see [4,5,6,7,8,9]) which treat the candidate data sets  as binary classification problems.
	Here the legitimate pulsar examples are a minority positive class, and spurious examples
	the majority negative class. At present multi-class labels are unavailable, given the
	costs associated with data annotation.
	
	The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639
	real pulsar examples. These examples have all been checked by human annotators. Each
	candidate is described by 8 continuous variables. The first four are simple statistics
	obtained from the integrated pulse profile (folded profile). This is an array of continuous
	variables that describe a longitude-resolved version of the signal that has been averaged
	in both time and frequency. The remaining four variables are
	similarly obtained from the DM-SNR curve. These are 
	summarised below:
	
	1. Mean of the integrated profile.
	2. Standard deviation of the integrated profile.
	3. Excess kurtosis of the integrated profile.
	4. Skewness of the integrated profile.
	5. Mean of the DM-SNR curve.
	6. Standard deviation of the DM-SNR curve.
	7. Excess kurtosis of the DM-SNR curve.
	8. Skewness of the DM-SNR curve.
	
	HTRU 2 Summary
	
	17,898 total examples.
	1,639 positive examples.
	16,259 negative examples.
