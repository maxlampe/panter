
//Script for recalculating DetSumX errors after uproot output
int recalcHistErr(const char* Sfile, const char* Ofile){

  //Import file
	TFile* fsourceS = new TFile(Sfile);
	if (!fsourceS->IsOpen()) {
		cerr << "Error: Cannot open file " << Sfile << endl;
	}

  std::cout << Sfile << std::endl;
  TH1D* detsum0;
  TH1D* detsum1;

  detsum0 = (TH1D*) fsourceS->Get("DetSum0");
  detsum1 = (TH1D*) fsourceS->Get("DetSum1");

  for(int i = 0; i < detsum0->GetNbinsX(); ++i){
    detsum0->SetBinError(i, TMath::Sqrt(TMath::Abs(detsum0->GetBinContent(i))));
  }
  for(int i = 0; i < detsum1->GetNbinsX(); ++i){
    detsum1->SetBinError(i, TMath::Sqrt(TMath::Abs(detsum1->GetBinContent(i))));
  }

  TFile* fout = new TFile(Ofile, "RECREATE");
  detsum0->Write();
  detsum1->Write();
  fout->Close();
  fsourceS->Close();

  gApplication->Terminate();
  return 0;
}
