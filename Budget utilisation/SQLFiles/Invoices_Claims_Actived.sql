SELECT invoices.member_id
     , invoices.state invoice_state
     , invoices.updated_at    
     , invoices.invoice_date
     , invoices.submitted_date
     , (invoiced_units * invoiced_unit_price) invoiced_amount
     , claim.invoice_id
     , claim.id claim_id
     , claim.start_date claim_start_date
     , claim.end_date claim_end_date     
     , claim.state claim_state
     , claim.funded_amount claim_funded_amount
     , claim.claimed_units
     , claim.claimed_unit_price
     , level3.key
     , level2.key
     , level1.key
  FROM HH_claim claim        
  INNER JOIN (
			   SELECT invoice.id
					, invoice.member_id
					, invoice.invoice_number
					, invoice.invoice_date
                    , invoice.submitted_date
				    , invoice.invoice_total
					, invoice.state invoice_state
					, invoice.funded_total
				    , invoice.is_reimbursement					
					, invoice.approved_date
					, invoice.received_date
				    , invoice.updated_at invoice_updated_at
					, invoice.provider_account_id
                    , invoice.state
                    , invoice.updated_at
                    , invoice.created_at
				 FROM HH_invoice invoice
			     INNER JOIN (
							  SELECT member.id member_id 
								   , MAX(plan.start_date) start_date
								   , plan.end_date
								FROM HH_member member
								INNER JOIN HH_plan plan
								        ON plan.member_key = member.member_key
                          
							 WHERE plan.status = 'PLAN_DELIVERY_ACTIVE'
							GROUP BY plan.member_key
						   ) plans ON plans.member_id = invoice.member_id 
								  AND (Date_Format(invoice.created_at,'%Y-%m-%d') >= Date_Format(plans.start_date,'%Y-%m-%d') AND Date_Format(invoice.created_at,'%Y-%m-%d') <= Date_Format(plans.end_date,'%Y-%m-%d'))
                                  
          ) invoices ON invoices.id = claim.invoice_id
                             
  INNER JOIN HH_item_category_level3 level3
          ON level3.id = claim.item_category_level3_id
          
  INNER JOIN HH_item_category_level2 level2
          ON level2.id = level3.item_category_level2_id
          
  INNER JOIN HH_item_category_level1 level1
          ON level1.id = level2.item_category_level1_id
		
    

   
