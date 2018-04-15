% Aum Sri Sai ram
load fisheriris.mat;
axis equal;
pause on;
K_no=2;
dimensions=3;
TOT_RUNS=30;
meas=meas(:,2:2+dimensions-1); %data set chosen
%meas=dlmread('wine.txt');
%meas=meas(:,2:5);
results=zeros(TOT_RUNS,4);
for run = 1:TOT_RUNS
a=PSO_func(meas);
%new_cent=[1.4936    0.2021;
 %   4.6477    1.5081];
 new_cent=a;
plot(meas(:,1),meas(:,2),'b*');
datasize=size(meas);
%Calculating distances for Kmeans
dist=zeros(datasize(1),K_no);
for z=1:90%no_of_iterations
   a=new_cent; 
for j= 1:K_no
    for k=1 : datasize(1)
        dist(k,j)=norm(a(j,:)-meas(k,:));
    end
end

        [value, index] = min(dist,[],2);
        d = index; %matrix d has the cluster indices to which it is closer
        
        hold on
for centroid = 1:K_no
   if any(d(:,1)==centroid)
     if (centroid==1)
       plot(meas(d(:,1)==centroid,1),meas(d(:,1)==centroid,2),'cO','Markersize',4);
     else if(centroid==2)
         plot(meas(d(:,1)==centroid,1),meas(d(:,1)==centroid,2),'mO','Markersize',4);
     end
   end
   end
end


new_cent=zeros(K_no,dimensions);
for centroid=1:K_no
    new_cent(centroid,1)=mean(meas((d(:)==centroid),1));
    new_cent(centroid,2)=mean(meas((d(:)==centroid),2));
end  
plot(meas(:,1),meas(:,2),'b*');


plot(a(1,1),a(1,2),'r*','MarkerSize',4);
plot(a(2,1),a(2,2),'r*','MarkerSize',4);


plot(new_cent(1,1),new_cent(1,2),'rO','MarkerSize',4);
plot(new_cent(2,1),new_cent(2,2),'rO','MarkerSize',4);
%pause(0.5);
end

%quantization error
averageK=zeros(1,K_no);
temp=0;
for j = 1 : K_no
    vectors_in_cluster=0;
    for i = 1:datasize(1)
        if (d(i) == j)
            vectors_in_cluster = vectors_in_cluster + 1;
            temp = temp + dist(i,j);
        end
    end
    averageK(1,j)=temp/vectors_in_cluster;
end
Q_error=sum(averageK)/K_no;

% intercluster
idx=d;

sumi=0;
for data=1: datasize(1)
   sumi = sumi + (norm(new_cent(idx(data),:)-meas(data,:)))^2;
end
inter=sumi/datasize(1);
i=1;
for centroids1 = 1 : K_no
 for centroids2 = 1 : K_no
    if(centroids1~=centroids2)
     intra(i)=norm(new_cent(centroids1,:)-new_cent(centroids2,:));
     i=i+1;
    end
 end
end
intra_cluster=min(intra);
%Accuracy
%correct=d(:,index);
meas_correct=dlmread('meas2D_truth.txt');
count_TP_TN=0;
for data=1:datasize(1)
    if (idx(data)==meas_correct(data))
        count_TP_TN=count_TP_TN+1;
    end
end
Accuracy=count_TP_TN/datasize(1);
results(run,1)=Q_error;
results(run,2) = inter;
results(run,3) = intra_cluster;
results(run,4) = Accuracy;
end
results

for centroid = 1:K_no
   if any(d(:)==centroid)
     if (centroid==1)
       plot(meas(d(:)==centroid,1),meas(d(:)==centroid,2),'cO','Markersize',4);
     else if(centroid==2)
         plot(meas(d(:)==centroid,1),meas(d(:)==centroid,2),'mO','Markersize',4);
     end
   end
   end
end
hold off